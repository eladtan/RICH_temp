from pathlib import Path
import xml.etree.ElementTree as ET
import json, math, re, subprocess, tarfile, io, shutil
root=Path('/home/maorm/RICH')
out=root/'build/branson_cp128'
src=out/'source/src'
# Recreate the benchmark source from the recorded clean upstream commit, not
# from the instrumented working tree. Only this task's archived files change.
commit=(root/'docs/branson_cp128/upstream_commit.txt').read_text().strip()
out.mkdir(parents=True,exist_ok=True)
(out/'source').mkdir(exist_ok=True)
archive=subprocess.check_output(['git','-C','/home/maorm/branson','archive',commit])
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
 tar.extractall(out/'source',filter='data')
if not (src/'pugixml/CMakeLists.txt').exists():
 shutil.copytree('/home/maorm/branson/src/pugixml',src/'pugixml',dirs_exist_ok=True)
p=src/'CMakeLists.txt';s=p.read_text()
old='set( CMAKE_CXX_FLAGS_RELEASE "-g -O3 -funroll-loops -fno-var-tracking-assignments")'
assert old in s
p.write_text(s.replace(old,'set( CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -march=x86-64-v3")'))
# Changes are confined to an archived, clean upstream checkout.
p=src/'mesh.h'; s=p.read_text(); old='cells[i].set_T_s(input.get_source_T());'
assert old in s
s=s.replace(old,'''// Crooked Pipe: drive only the thin inlet material (region 1).
        cells[i].set_T_s(region_ID == 1 ? input.get_source_T() : 0.0);''')
p.write_text(s)
p=src/'constants.h';s=p.read_text();assert 'cutoff_fraction = 0.01;' in s
p.write_text(s.replace('cutoff_fraction = 0.01;', 'cutoff_fraction = 0.001;'))
# Lightweight temperature probes outside the timed transport loop; MPI invocation
# timing includes them, as it includes STORM's probe output.
p=src/'particle_pass_driver.h';s=p.read_text();needle='    mesh.update_temperature(abs_E, track_E, imc_state);'
assert needle in s
s=s.replace(needle,needle+'''
    {
      const double px[5] = {0.25, 2.75, 3.5, 4.25, 6.75};
      const double pr[5] = {0.0, 0.0, 1.25, 0.0, 0.0};
      double values[10] = {};
      for (const auto &cell : mesh) {
        if (cell.get_region_ID() != 1) continue;
        float xyz[3]; cell.get_center(xyz);
        const double r = std::sqrt(double(xyz[1])*xyz[1] + double(xyz[2])*xyz[2]);
        for (int k=0; k<5; ++k) {
          const double dx=xyz[0]-px[k], dr=r-pr[k];
          if (dx*dx + dr*dr <= 0.01) {
            values[k] += cell.get_T_e()*cell.get_volume();
            values[k+5] += cell.get_volume();
          }
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, values, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      if (rank == 0) {
        std::cout << "CP_PROBE " << imc_state.get_step() << " "
                  << (imc_state.get_time()+imc_state.get_dt())*10.0;
        for (int k=0; k<5; ++k) std::cout << " " << (values[k+5]>0 ? values[k]/values[k+5] : -1.0);
        std::cout << std::endl;
      }
    }
''')
p.write_text(s)
prototype=ET.Element('prototype')
def child(p,k,v):ET.SubElement(p,k).text=str(v)
c=ET.SubElement(prototype,'common')
dt=1e-11;t=0
for i in range(120):t+=dt;dt=min(dt*1.1,1e-9)
settings={'method':'IMC','t_start':0,'t_stop':format(t/1e-8,'.17g'),'dt_start':.001,'t_mult':1.1,'dt_max':.1,'photons':1858414,'seed':14706,'use_combing':'FALSE','use_gpu_transporter':'FALSE','dd_transport_type':'PARTICLE_PASS','n_omp_threads':1,'mesh_decomposition':'METIS','dd_batch_size':10000,'particle_message_size':10000,'particle_storage':'SOA','particle_algorithm':'HISTORY','output_frequency':1,'write_silo':'FALSE'}
for k,v in settings.items():child(c,k,v)
d=ET.SubElement(prototype,'debug_options')
for k in ['print_verbose','print_mesh_info']:child(d,k,'FALSE')
s=ET.SubElement(prototype,'spatial')
xd=[(0,2.5,40),(2.5,3,8),(3,4,16),(4,4.5,8),(4.5,7,40)]
yd=[(-2+i*.08,-2+(i+1)*.08,1) for i in range(50)]
for axis,div in [('x',xd),('y',yd),('z',yd)]:
 for lo,hi,n in div:
  e=ET.SubElement(s,axis+'_division')
  for k,v in [(axis+'_start',lo),(axis+'_end',hi),('n_'+axis+'_cells',n)]:child(e,k,v)
def thick(x,y,z):
 r=math.hypot(y,z)
 if r>1.5:return True
 if r<.5:return 3<x<4
 if x<2.5 or x>4.5:return True
 return 3<x<4 and r<1
thinvolume=0;inletarea=0;thin_cells=0
for ix,(xl,xh,nx) in enumerate(xd):
 for iy,(yl,yh,_) in enumerate(yd):
  for iz,(zl,zh,_) in enumerate(yd):
   th=thick((xl+xh)/2,(yl+yh)/2,(zl+zh)/2)
   e=ET.SubElement(s,'region_map')
   for k,v in [('x_div_ID',ix),('y_div_ID',iy),('z_div_ID',iz),('region_ID',2 if th else 1)]:child(e,k,v)
   if not th:
    thin_cells+=nx;thinvolume+=(xh-xl)*(yh-yl)*(zh-zl)
    if ix==0:inletarea+=(yh-yl)*(zh-zl)
b=ET.SubElement(prototype,'boundary')
for face in ['left','right','up','down','top','bottom']:child(b,'bc_'+face,'SOURCE' if face=='left' else 'VACUUM')
child(b,'T_source',.5)
r=ET.SubElement(prototype,'regions')
for rid,rho,cv,opac in [(1,2,.0005,.2),(2,1,1,2000)]:
 e=ET.SubElement(r,'region')
 for k,v in {'ID':rid,'density':rho,'CV':cv,'opacA':opac,'opacB':0,'opacC':0,'opacS':0,'initial_T_e':.05,'initial_T_r':.05}.items():child(e,k,v)
ET.indent(prototype)
ET.ElementTree(prototype).write(out/'crooked_pipe.xml',encoding='unicode')
meta={'mesh':[112,50,50],'cells':280000,'thin_cells':thin_cells,'thin_volume_cm3':thinvolume,'analytic_thin_volume_cm3':4.75*math.pi,'inlet_area_cm2':inletarea,'analytic_inlet_area_cm2':math.pi*.25,'end_time_s':t,'cycles':120,'nominal_photons_per_step':1858414,'budget_basis':'rounded mean STORM prestep_generated count over 120 steps of accepted log1; native allocation and carried census differ','initial_radiation_note':'Branson initializes radiation packets at 0.05 keV; existing STORM log starts with zero census packets despite initializing cell Erad to 0.05 keV equilibrium'}
(out/'configuration.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))

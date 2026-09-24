from pathlib import Path
import json,shlex,subprocess,sys
root=Path('/home/maorm/RICH'); work=root/'build/rdma128_followup'; build=root/'build/rdma128_20260912/candidate_build'
name=sys.argv[1]; flags=sys.argv[2:]
e=next(x for x in json.loads((build/'compile_commands.json').read_text()) if x['file'].endswith('crooked_pipe/main.cpp'))
cmd=shlex.split(e['command']);cmd[cmd.index('-o')+1]=str(work/(name+'.o'));cmd+=flags
subprocess.run(cmd,cwd=e['directory'],check=True)
cmd=shlex.split((build/'examples/CMakeFiles/crooked_pipe.dir/link.txt').read_text());cmd[cmd.index('-o')+1]=str(work/name)
cmd=[str(work/(name+'.o')) if x=='CMakeFiles/crooked_pipe.dir/crooked_pipe/main.cpp.o' else x for x in cmd]
subprocess.run(cmd,cwd=build/'examples',check=True)

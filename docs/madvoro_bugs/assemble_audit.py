#!/usr/bin/env python3
"""Assemble the editable audit from saved, reviewed source notes."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
NOTES = ROOT / 'audit_notes'

def section(filename, prefix):
    text = (NOTES / filename).read_text()
    match = re.search(r'^## ' + re.escape(prefix) + r'[^\n]*\n(.*?)(?=^## |\Z)', text, re.M | re.S)
    if not match:
        raise ValueError((filename, prefix))
    return match.group(1).strip()

groups = [
    ('4. Partial builds: linked defects that require a coherent repair', [
        ('B01 / P1: inactive generators are deleted', 'root_findings.md', 'B01:'),
        ('B02 / P1: all-point tree reads active-point storage', 'root_findings.md', 'B02:'),
        ('B03 / P1: reversed active-to-all map', 'root_findings.md', 'B03:'),
        ('B04 / P1 conditional: one-way ghost dependencies are erased', 'mpi.md', 'MPI-02:'),
    ]),
    ('5. Copy, build mode and lifecycle errors', [
        ('B05 / P1: built copies have null query trees', 'geometry.md', 'Finding G-B01'),
        ('B06 / P2: unbuilt copies dereference a null balancer', 'geometry.md', 'Finding G-B02'),
        ('B07 / P1: mixed suppression selects stale weight storage', 'integration.md', 'INT-01:'),
        ('B08 / P2: serial Build selects distributed state', 'root_findings.md', 'B08:'),
        ('B09 / P2: first suppressed build skips initialization', 'root_findings.md', 'B09:'),
        ('B10 / P2: releasing memory changes a custom domain', 'geometry.md', 'Finding G-B03'),
        ('B11 / P2: continuity query indexes an empty mesh', 'root_findings.md', 'B11:'),
        ('B12 / P2: kernel replacement cannot rebuild its converter', 'integration.md', 'INT-02:'),
    ]),
    ('6. Periodic identity and geometric helper errors', [
        ('B13 / P2: remote image resolves to an unrelated local point', 'root_findings.md', 'B13:'),
        ('B14 / P2: mixed-periodic image queries are duplicated', 'mpi.md', 'MPI-03:'),
        ('B15 / P2, source-only: face cleanup retries with stale state', 'geometry.md', 'Finding G-B04'),
        ('B16 / P3: standalone face centroid lacks area normalization', 'geometry.md', 'Finding G-B05'),
    ]),
    ('7. Shared dependencies: active and configuration-specific defects', [
        ('D01 / P1 conditional: request completion compaction', 'mpi.md', 'MPI-01:'),
        ('D02 / P1: manager and balancer use different communicators', 'integration.md', 'INT-03:'),
        ('D03 / P2: one-rank 1D bins violate the query invariant', 'integration.md', 'INT-04:'),
        ('D04 / P2: buffer assembly exceeds receive capacity', 'mpi.md', 'DEP-01:'),
        ('D05 / P2: eight-byte payload never dispatches', 'mpi.md', 'DEP-02:'),
        ('D06 / P2: flattened query results lose self answers', 'mpi.md', 'DEP-03:'),
    ]),
    ('8. RICH integration: physical-field transfer errors outside MadVoro', [
        ('R01 / P1: chain transfer ignores destination indices', 'integration.md', 'INT-05:'),
        ('R02 / P1: empty-origin rank skips field transfer', 'integration.md', 'INT-06:'),
    ]),
]

chunks = [(NOTES / 'report_front.md').read_text()]
for heading, items in groups:
    chunks.append('\n## ' + heading + '\n')
    if heading.startswith('7.'):
        chunks.append('D01 is used by the active ghost-query transport. D02-D06 have concrete component failures under the stated communicator, backend, payload or option; ordinary current MadVoro use of those exact configurations was not established. Each repair belongs in its owning dependency, followed by a reviewed submodule-pointer update.\n')
    if heading.startswith('8.'):
        chunks.append('These two errors are physically in RICH, not the MadVoro submodule. They are included because a correct mesh and correct ownership map are insufficient if application fields are transferred to the wrong slots or ranks fail to participate.\n')
    for title, filename, prefix in items:
        content = section(filename, prefix)
        chunks.append('\n### ' + title + '\n\n' + content + '\n')
chunks.append((NOTES / 'report_handoff.md').read_text())
text = '\n'.join(chunks)
for old, new in {'G-B01':'B05','G-B02':'B06','G-B03':'B10','G-B04':'B15','G-B05':'B16'}.items():
    text = text.replace(old, new)
text = text.replace('[MPI Forum multiple-completion contract](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node76.htm)', '[MPI Forum multiple-completion contract](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf)')
# Remove investigation-coordination language that is irrelevant to a final handoff.
text = text.replace("the root audit's earlier tree/subset defects currently obstruct a trustworthy full partial build", "B01-B03 currently obstruct a trustworthy full partial build")
text = text.replace('Coordinate with the root partial-build fix', 'Coordinate with B01-B03')
text = text.replace('A clean transport rerun is also retained if present.', 'The logged rank-local stages and source branch, rather than transport warnings, establish the mismatch.')
text = text.replace('This note does not claim', 'This report does not claim')
text = text.replace('on MPI_COMM_SELF on each rank', 'on `MPI_COMM_SELF` on each rank')
text = text.replace('throws unless bins_.size()==MPI size.', 'throws unless `bins_.size()` equals the communicator size.')
for old, new in {
    'processes0': 'processes 0', 'slot2': 'slot 2', 'slot0': 'slot 0',
    'to2 entries': 'to 2 entries', 'index2': 'index 2', 'a2-entry': 'a 2-entry',
    'including0': 'including 0', 'arrays0–8': 'arrays 0–8',
}.items():
    text = text.replace(old, new)
(ROOT / 'MadVoro_Bug_Audit.md').write_text(text)
print(f'{len(text.split())} words; {sum(len(x[1]) for x in groups)} findings')

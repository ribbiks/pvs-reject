# pvs-reject
Generates reject tables for DooM maps from subsector potentially visible sets (PVS).

This code is essentially a python rewrite of Jānis Legzdiņš' [glVIS](https://github.com/TriggerCoder/glvis) application, with a few modifications:
* Support for maps with up to 65535 lines
* Blunt parallelization for processing large maps

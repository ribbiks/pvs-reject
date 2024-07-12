# pvs-reject
Generates reject tables for DooM maps from subsector potentially visible sets (PVS).

This code is essentially a dumbed-down python rewrite of iD Software's 'vis' utility for Quake, adapted to Doom maps following the example of [glVIS](https://github.com/TriggerCoder/glvis), but with a few modifications:
* Support for maps with up to 65535 lines
* Blunt parallelization for processing large maps

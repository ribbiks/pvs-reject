# pvs-reject
Generates reject tables for DooM maps from subsector potentially visible sets (PVS).

This code is essentially a dumbed-down python rewrite of iD Software's 'vis' utility for Quake, adapted to Doom maps following the example of [glVIS](https://github.com/TriggerCoder/glvis).

Disclaimer: This is a failed experiment. It works fine on very small maps but the performance scales terribly and is not usable on anything sizable.

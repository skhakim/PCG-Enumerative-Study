Artifact for "SMT over Tree Metrics: Enumerating Pairwise Compatibility Graphs".

Contents:
- pcg_library_star_cat_gen.py: exact SMT checker for star-PCG, caterpillar-PCG, and general PCG.
- classify_pcg.py: parallel enumeration driver using nauty geng.
- results/vert_8, results/vert_9, results/vert_10: JSONL outputs and summary files.

Main results:
- n=8: 11,117 connected unlabeled graphs classified in 50s.
- n=9: 261,080 connected unlabeled graphs classified in 32m35s.
- n=10: 11,716,571 connected unlabeled graphs classified in 170h21m21s.

Hardware for n=10:
CloudLab Clemson ibm8335 node, 160 workers.

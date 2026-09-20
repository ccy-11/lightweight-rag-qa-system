# Reproducibility Checklist

- [ ] Random seed recorded
- [ ] All library versions recorded (torch, transformers, faiss-cpu, sentence-transformers, rouge-score)
- [ ] Hardware environment recorded (GPU model / CPU / RAM)
- [ ] 4-bit quantization params (nf4) recorded
- [ ] Greedy decoding (do_sample=False) confirmed
- [ ] Index file path recorded
- [ ] QA dataset source recorded (dataset_id + download command)
- [ ] Experiment script can be re-run (python scripts/run_eval.py --seed {seed})

## Feature selection for ICL algorithms

### Installation

* install tools - `pip install .`
* install scikit-feature - `cd scikit-feature && python3 setup.py install`

### Run code (without ICL)

```bash
python3 fs_without_icl.py -env gridworld -save_dir save/gridworld -seed 1
```

We can change the `-env` and `-save_dir` accordingly. Possible values are `gridworld`, `cartpole`, `highd`, `ant`, `hc` (HalfCheetah). We can use the value `cartpoletest` for the modified Cartpole environment with the bogus feature.

For HighD-S, use
```
python3 fs_without_icl.py -env highd -save_dir save/highds -seed 1 -reduced
```

## Run code (with ICL)

Use the same arguments as without ICL, but use the file `fs_with_icl.py`.

## Run code (with baseline)

```bash
python3 fs_baseline.py -env gridworld -seed 1 -baseline cife
```

Here, the `-env` value can be changed as per previous instructions. The `-baseline` argument can have 4 values: `cife`, `disr`, `cmim`, `mrmr`.
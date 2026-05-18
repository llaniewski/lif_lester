# LIF model implemented in [L3STER](https://github.com/kubagalecki/L3STER)

## On Athena
Get code
```bash
git clone git@github.com:llaniewski/lif_lester.git
cd lif_lester
git submodule init
get submodule update
```
Install spack and dependencies
```bash
sbatch -A [grant] -p [partition] athena/install.sh
```
Compile
```bash
sbatch -A [grant] -p [partition] athena/compile.sh
```
Run
```bash
sbatch -A [grant] -p [partition] athena/run.sh example/light_box_3d.xml
```

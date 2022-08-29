# PlotNeuralNet

Basaed on [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2526396.svg)](https://doi.org/10.5281/zenodo.2526396).
Latex code for drawing neural networks for reports and presentation on Ubuntu 18.04.6 LTS. Applied some minor changes to improve its usability (see botom of the README.md). 

## Examples
Following are some network representations:

# Example of use 
0. Install git and latex packages

```
sudo apt install git
sudo apt-get install texlive-latex-base
sudo apt-get install texlive-fonts-recommended
sudo apt-get install texlive-fonts-extra
sudo apt-get install texlive-latex-extra
```
        
1. Clone this repository and navigate to the path where the plot tool is.
  ```
  cd ~/
  git clone https://github.com/AUROVA-LAB/aurova_machine_learning.git
  cd aurova_machine_learning/Drawing_tools/PlotNeuralNet/
  ```

2. Execute the example as followed.
  ```
  cd pyexamples/
  bash ../tikzmake.sh test_simple
  ```
    
3. In order to complete with texts the scheme, an external pdf editor is needed (tested Adobe Acrobat DC).


## TODO

- [ ] Add examples.


## Python usage

First, navigate into ~/aurova_machine_learning/Drawing_tools/PlotNeuralNet/pyexamples/ and create a new Python file:

    cd ~/aurova_machine_learning/Drawing_tools/PlotNeuralNet/pyexamples/
    gedit new_scheme.py

Add the following code to your new file:

```python
import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2, colour="{rgb:yellow,1.0;red,2.5;white,5}" ),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
    to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2, colour="{rgb:brown,1.0;green,3.0;white,5}" ),
    to_connection( "pool1", "conv2", origin="-east", destination="-west" ),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
    to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
    to_connection("pool2", "soft1", origin="-east", destination="-west" ),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
```

Now, run the program as follows:

    bash ../tikzmake.sh my_arch

## Applied changes
1. Added the option to choose the colour of each convolution block. It provides more customization. In file ~/aurova_machine_learning/Drawing_tools/PlotNeuralNet/pycore/tikzeng.py the function to_Conv is changed from to_Conv( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" ") to to_Conv( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" ", colour="{rgb:yellow,5;red,2.5;white,5}" ).

2. Added the option to choose the origin and goal point (north, south, east or west) of the block as source and goal. In file ~/aurova_machine_learning/Drawing_tools/PlotNeuralNet/pycore/tikzeng.py the function to_connection is changed from to_connection( of, to) to to_connection( of, to, origin="-east", destination="-west").

3. Avoided that the program deletes the .tex generated file. In file ~/aurova_machine_learning/Drawing_tools/PlotNeuralNet/tikzmake.sh a line is commented (#rm *.tex).

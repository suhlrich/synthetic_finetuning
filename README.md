create environment
`conda create --name synthetic-finetuning python=3.9`

install cuda toolkit that matches your cuda install
`conda install cudatoolkit=11.1 -c pytorch -c conda-forge`

install requirements
`conda install --file requirements.txt -c conda-forge -c pytorch`
`pip install -r requirements-pip.txt`

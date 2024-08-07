# TTS tuto

## Setup Instructions

1. Start by instanciating your Conda environment by running the following command:

    ```sh
    conda env create -v -f environment.yml --force
    conda activate tts_playground
    ```

2. Install project dependencies with Poetry:

    ```sh
    poetry install
    ```

3. Check current environment health:

    ```sh
    poetry check
    # >>> All set!
    ```

4. Declare any additional dependency automatically in the `pyproject.tmol` file by running the command:

    ```sh
    poetry add <pachage>
    poetry add <pachage>@<^version>  # If version is 2.0.5 then allow >=2.0.5, <3.0.0 versions
    poetry add "<pachage>>=<version>"  # No upper bound
    poetry add <pachage>==<version>  # Allow only selected version
    ```

    ex:

    ```sh
    poetry add numpy
    poetry add torch torchvision torchaudio --source pytorch
    ```

    ⚠️ torch issue `OSError: [WinError 126]` with `tts_playground\Lib\site-packages\torch\lib\fbgemm.dll`?

    💡 Solution: Download `libomp140.x86_64.dll` and add it to the same folder as `fbgemm.dll`

## Project dependencies

To run this project properly you will need:

- [Miniconda installed](https://docs.anaconda.com/free/miniconda)

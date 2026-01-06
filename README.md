# Machine hearing project
### Using CARFAC ear simulator and HTM from Numenta

### Short descrption

## Prerequisites
```
brew install cmake
brew install fluidsynth
brew install opencv
brew install fftw
brew install asio
```

### Build eigen library
```
git clone https://gitlab.com/libeigen/eigen.git --branch 3.4.0 --depth 1
cd eigen && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../dist && make -j16 install
cd ..
cp -r dist ../ear_model/eigen
```

### Build htm library
```
git clone https://github.com/htm-community/htm.core htm --branch master --depth 1
cd htm
brew install pybind11
python3 -m venv .venv
source .venv/bin/activate
python htm_install.py
cp -r dist/Release ../ear_model/htm
```

### Download sound fonts (1GB)
```
cd ear_model
mkdir sound_fonts && cd sound_fonts
curl -L --output sound_font.sf2 "https://huggingface.co/datasets/projectlosangeles/soundfonts4u/resolve/main/Essential%20Keys-sforzando-v9.6.sf2?download=true"
```

## Training setup
### Generate dataset
```
cd build
cmake .. && make -j16 && ./midi_gen
```
### Test playback
```
cmake .. && make -j16 && ./playback
```
### Launch tranining
```
cmake && make -j16 && ./tbt_train
```
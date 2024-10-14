# git clone git@github.com:MSallermann/mumax3.git --branch no_wrappers
# cd mumax3
go mod init github.com/Msallermann/mumax3 # This creates a go.mod file
echo "replace github.com/mumax/3 => $(pwd)" >> go.mod
go mod tidy
export NVCC_CCBIN=gcc-8 # If the gcc on your path is version > 8
export GOBIN=$(pwd)/bin # Change this to whatever path you want
export CUDA_CC=75
# export NVCC_CCBIN=gcc-13 # For some reason I have to specifically set this ... 
make realclean && make # Have to rebuild the cuda kernels
go install ./cmd/mumax3
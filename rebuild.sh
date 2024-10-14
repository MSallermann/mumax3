export GOBIN=$(pwd)/bin # Change this to whatever path you want
make realclean && make # Have to rebuild the cuda kernels
go install ./cmd/mumax3 # Go install is the recommended way to install commands now
package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func Zero3(m *data.Slice) {
	
	util.Assert(m.NComp() == 3)
	
	N := m.Len()
	cfg := make1DConf(N)

	k_zero3_async(
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		N, cfg)
}

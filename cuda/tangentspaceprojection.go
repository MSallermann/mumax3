package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func TangentSpaceProjection(k, m *data.Slice) {
	util.Argument(k.NComp() == 3 && m.NComp() == 3)
	util.Argument(k.Len() == m.Len())

	N := k.Len()
	cfg := make1DConf(N)
	k_tangentspaceprojection_async(
		k.DevPtr(X), k.DevPtr(Y), k.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		N, cfg)
}

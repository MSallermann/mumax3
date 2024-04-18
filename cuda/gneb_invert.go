package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// dst += prefactor * dot(a, b), as used for energy density
func Invert(dst, src *data.Slice) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Argument(dst.Len() == src.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_invert_async(dst.DevPtr(0), src.DevPtr(0), N, cfg)
}

package main

import (
	"fmt"
	"github.com/clyfar/nist80022"
)

func main() {
	data := []int{33, 460, 340, 126, 468, 359, 348, 6, 376, 325, 52, 15, 307, 431, 389, 230, 32, 393, 461, 416, 302, 26, 422, 424, 217, 349, 463, 309, 100, 69, 266, 61, 214, 12, 396, 420, 366, 152, 93}

	ndata := nist80022.IntsToBits(data)
	fmt.Println(nist80022.CumulativeSumsTest(ndata))
	fmt.Println(nist80022.MonobitTest(ndata))
	fmt.Println(nist80022.BlockFrequencyTest(ndata, 32))
	fmt.Println(nist80022.LongestRunOfOnesTest(ndata, 8))
	fmt.Println(nist80022.BinaryMatrixRankTest(ndata, 16, 16))
	fmt.Println(nist80022.RunsTest(ndata))
	fmt.Println(nist80022.SpectralTest(ndata))
	fmt.Println(nist80022.UniversalStatisticalTest(ndata))
}

package nist80022

import (
	"fmt"
	"math"
	"math/cmplx"
	"strings"

	"gonum.org/v1/gonum/dsp/fourier"
	"gonum.org/v1/gonum/mathext"
	"gonum.org/v1/gonum/stat/distuv"
)

// countOnes returns the number of ones in a binary string
func countOnes(bits string) int {
	count := 0
	for _, b := range bits {
		if b == '1' {
			count++
		}
	}
	return count
}

// average returns the average of a slice of floats
func average(nums []float64) float64 {
	if len(nums) == 0 {
		return 0
	}
	sum := 0.0
	for _, x := range nums {
		sum += x
	}
	return sum / float64(len(nums))
}

// variance returns the variance of a slice of floats, given its average
func variance(nums []float64, avg float64) float64 {
	if len(nums) == 0 {
		return 0
	}
	sum := 0.0
	for _, x := range nums {
		sum += (x - avg) * (x - avg)
	}
	return sum / float64(len(nums))
}

// IntsToBits is a function to convert integers to binary and concatonate the results
func IntsToBits(ints []int) string {
	var bits strings.Builder
	for _, i := range ints {
		bits.WriteString(fmt.Sprintf("%b", i))
	}
	return bits.String()
}

// chiSquared calculates the chi-squared statistic for two sets of counts and expected counts
func chiSquared(counts []float64, expectedCounts []float64) float64 {
	chisq := 0.0
	for i, count := range counts {
		expected := expectedCounts[i]
		chisq += (count - expected) * (count - expected) / expected
	}
	return chisq
}

// chiSquaredCDF is an implementation of the chi-squared cumulative distribution function
func chiSquaredCDF(chisq float64, df int) float64 {
	chiDist := distuv.ChiSquared{K: float64(df)}
	pval := 1 - chiDist.CDF(chisq)
	return pval
}

// BinaryStringToIntSlice should be removed if not needed in the future
func BinaryStringToIntSlice(str string) ([]int, error) {
	intSlice := make([]int, len(str))
	for i, char := range str {
		if char == '0' {
			intSlice[i] = 0
		} else if char == '1' {
			intSlice[i] = 1
		} else {
			return nil, fmt.Errorf("invalid character in binary string: %c", char)
		}
	}
	return intSlice, nil
}

// getBlock returns a block of size len(s) starting at startIndex
func getBlock(s string, startIndex int) string {
	if startIndex+len(s) > len(s) {
		return ""
	}
	return s[startIndex : startIndex+len(s)]
}

// BitArray is a structure representing an array of bits.
type BitArray struct {
	data []byte // data holds the actual bytes storing the bits.
}

// NewBitArray initializes and returns a new BitArray of the specified size.
func NewBitArray(size int) *BitArray {
	return &BitArray{
		data: make([]byte, (size+7)>>3), // Allocate the byte array, taking into account the size of a byte (8 bits).
	}
}

// Set modifies the value of the bit at the specified index to the given value (true or false).
func (b *BitArray) Set(index int, value bool) {
	if value {
		b.data[index>>3] |= 1 << (index & 7) // Set the bit to 1.
	} else {
		b.data[index>>3] &^= 1 << (index & 7) // Set the bit to 0.
	}
}

// Get retrieves the boolean value of the bit at the specified index.
func (b *BitArray) Get(index int) bool {
	return (b.data[index>>3] & (1 << (index & 7))) != 0 // Extract the bit value and convert it to a boolean.
}

// Size returns the total number of bits in the BitArray.
func (b *BitArray) Size() int {
	return len(b.data) << 3 // Calculate the number of bits by multiplying the length of the byte array by 8.
}

// IntArrayToBitArray converts an array of integers into a BitArray.
func IntArrayToBitArray(intArray []int) *BitArray {
	n := len(intArray) * 8     // Calculate the total number of bits required to store the integers.
	bitArray := NewBitArray(n) // Create a new BitArray of the required size.

	// Iterate through the integers and set their bits in the BitArray.
	for i, value := range intArray {
		for j := 0; j < 8; j++ {
			bit := (value >> (7 - j)) & 1 // Extract the individual bits from the integer.
			bitArray.Set(i*8+j, bit == 1) // Set the corresponding bit in the BitArray.
		}
	}

	return bitArray
}

// Perform Gaussian elimination to compute the rank of the matrix
func matrixRank(matrix [][]int, size int) int {
	rank := 0

	for row := 0; row < size; row++ {
		if matrix[row][row] == 0 {
			found := false
			for i := row + 1; i < size; i++ {
				if matrix[i][row] != 0 {
					matrix[row], matrix[i] = matrix[i], matrix[row]
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}

		rank++

		for i := row + 1; i < size; i++ {
			if matrix[i][row] != 0 {
				for j := row; j < size; j++ {
					matrix[i][j] = (matrix[i][j] + matrix[row][j]) % 2
				}
			}
		}
	}

	return rank
}

// BinaryMatrixRankTest performs the binary matrix rank test on the input
// bit string. The test divides the bit string into m x q binary matrices,
// and calculates the rank of each matrix. It then calculates the chi-squared
// statistic from the deviation of the observed rank sum from the expected rank
// sum, and computes the corresponding p-value from the gamma distribution.
func BinaryMatrixRankTest(bitArray *BitArray, matrixSize int) (float64, bool) {
	n := bitArray.Size()
	nMatrices := n / (matrixSize * matrixSize)
	matrixSizeF := float64(matrixSize)

	pFullRank := 1 - math.Pow(1-math.Pow(2, -matrixSizeF), matrixSizeF)
	pOneLessRank := math.Pow(1-math.Pow(2, -matrixSizeF), matrixSizeF) - math.Pow(1-math.Pow(2, -matrixSizeF), matrixSizeF-1)
	pDeficientRank := 1 - pFullRank - pOneLessRank

	fullRankCount := 0
	oneLessRankCount := 0
	btoi := 0

	for i := 0; i < nMatrices; i++ {
		matrix := make([][]int, matrixSize)
		for row := 0; row < matrixSize; row++ {
			matrix[row] = make([]int, matrixSize)
			for col := 0; col < matrixSize; col++ {
				index := i*matrixSize*matrixSize + row*matrixSize + col
				if bitArray.Get(index) {
					btoi = 1
				} else {
					btoi = 0
				}
				matrix[row][col] = btoi
			}
		}

		rank := matrixRank(matrix, matrixSize)

		if rank == matrixSize {
			fullRankCount++
		} else if rank == matrixSize-1 {
			oneLessRankCount++
		}
	}

	chiSquare := math.Pow(float64(fullRankCount)-float64(nMatrices)*pFullRank, 2) / (float64(nMatrices) * pFullRank)
	chiSquare += math.Pow(float64(oneLessRankCount)-float64(nMatrices)*pOneLessRank, 2) / (float64(nMatrices) * pOneLessRank)
	chiSquare += math.Pow(float64(nMatrices-fullRankCount-oneLessRankCount)-float64(nMatrices)*pDeficientRank, 2) / (float64(nMatrices) * pDeficientRank)
	pValue := math.Exp(-chiSquare / 2)

	alpha := 0.001
	pass := pValue > alpha
	return pValue, pass
}

// CumulativeSumsTest or the NIST cumulative sums test is a statistical test to determine if the number of ones and zeros in a binary sequence are evenly distributed.
// The test involves calculating the cumulative sums of the deviations between the observed counts of ones and zeros and the expected counts.
// If the sequence passes the test, then it is considered to be statistically random.
func CumulativeSumsTest(data *BitArray) (float64, bool) {
	n := data.Size()

	if n < 100 {
		return -1.0, false
	}

	S := make([]int, n+1)
	S[0] = 0
	for i := 1; i <= n; i++ {
		if data.Get(i - 1) {
			S[i] = S[i-1] + 1
		} else {
			S[i] = S[i-1] - 1
		}
	}

	zForward := math.Abs(float64(S[n])) / math.Sqrt(float64(n))
	zBackward := math.Abs(float64(S[0])) / math.Sqrt(float64(n))

	pValueForward := 1.0 - math.Erf(zForward/(math.Sqrt(2.0)*math.Sqrt(2.0)))
	pValueBackward := 1.0 - math.Erf(zBackward/(math.Sqrt(2.0)*math.Sqrt(2.0)))

	alpha := 0.001

	return pValueForward, pValueForward > alpha && pValueBackward > alpha
}

// MonobitTest performs the frequency (monobit) test on the input sequence.
// The input sequence should be a string, where each character represents
// the value of a bit (either '0' or '1').
// The function returns a boolean value indicating whether the input
// sequence passes the test (true) or fails the test (false).
func MonobitTest(bitArray *BitArray) (float64, bool) {
	sqrt2 := 1.41421356237309504880
	n := bitArray.Size()
	ones := 0

	for i := 0; i < n; i++ {
		if bitArray.Get(i) {
			ones++
		}
	}

	zeros := n - ones
	S := ones - zeros
	sObs := float64(S) / math.Sqrt(float64(n))

	pValue := math.Erfc(math.Abs(sObs) / sqrt2)
	alpha := 0.001
	pass := pValue >= alpha

	return pValue, pass
}

// BlockFrequencyTest performs the block frequency test on the input sequence.
// The input sequence should be a slice of integers, where each integer
// represents the value of a bit (either 0 or 1)
// The function returns a boolean value indicating whether the input
// sequence passes the test (true) or fails the test (false).
func BlockFrequencyTest(bitArray *BitArray, blockSize int) (float64, bool) {
	n := bitArray.Size()
	nBlocks := n / blockSize

	chiSquare := 0.0

	for i := 0; i < nBlocks; i++ {
		ones := 0

		for j := 0; j < blockSize; j++ {
			if bitArray.Get(i*blockSize + j) {
				ones++
			}
		}

		pi := float64(ones) / float64(blockSize)
		chiSquare += (pi - 0.5) * (pi - 0.5)
	}

	chiSquare *= 4.0 * float64(blockSize)
	pValue := mathext.GammaIncReg(float64(nBlocks)/2.0, chiSquare/2.0)

	alpha := 0.001
	pass := pValue >= alpha
	return pValue, pass
}

// RunsTest performs the runs test on the input bit string, which tests for
// the presence of long runs of consecutive zeros or ones in the sequence.
// The test counts the total number of runs in the sequence, and calculates
// the expected value and variance of this count under the assumption of
// randomness. It then computes the z-score of the observed run count, and
// computes the corresponding p-value from the standard normal distribution.
// The function returns the negative base 10 logarithm of the p-value as the score.
func RunsTest(data *BitArray) (float64, bool) {
	n := data.Size()

	if n < 100 {
		return -1.0, false
	}

	var n1, n2, runs int
	for i := 0; i < n; i++ {
		if data.Get(i) {
			n1++
		} else {
			n2++
		}
		if i == 0 || data.Get(i) != data.Get(i-1) {
			runs++
		}
	}

	p := float64(n1) / float64(n)
	q := float64(n2) / float64(n)
	tau := 2.0 / math.Sqrt(float64(n))

	if math.Abs(p-q) >= tau {
		return 0.0, false
	}

	expectedRuns := 2.0*float64(n1)*float64(n2)/float64(n) + 1.0
	varianceRuns := 2.0 * float64(n1) * float64(n2) * (2.0*float64(n1)*float64(n2) - float64(n)) / (math.Pow(float64(n), 2.0) * (float64(n) - 1.0))

	z := float64(float64(runs)-expectedRuns) / math.Sqrt(varianceRuns)
	pValue := math.Erfc(math.Abs(z) / math.Sqrt(2.0))
	alpha := 0.001

	return pValue, pValue > alpha
}

// LongestRunOfOnesTest performs the longest run of ones in a block test on the input sequence.
// The function returns a boolean value indicating whether the input
// sequence passes the test (true) or fails the test (false).
func LongestRunOfOnesTest(data *BitArray) (float64, bool) {
	n := data.Size()
	if n < 128 {
		return -1.0, false
	}

	blockLength := 0
	if n < 6272 {
		blockLength = 8
	} else if n < 750000 {
		blockLength = 128
	} else {
		blockLength = 10000
	}

	numBlocks := int(math.Floor(float64(n) / float64(blockLength)))

	// Count the frequencies of runs of ones of various lengths
	nu := make([]int, 6)
	for i := 0; i < numBlocks; i++ {
		runLength := 0
		maxRunLength := 0
		for j := 0; j < blockLength; j++ {
			bit := data.Get(i*blockLength + j)
			if bit {
				runLength++
			} else {
				if runLength > 0 {
					if runLength > maxRunLength {
						maxRunLength = runLength
					}
					runLength = 0
				}
			}
		}

		if maxRunLength <= 1 {
			nu[0]++
		} else if maxRunLength == 2 {
			nu[1]++
		} else if maxRunLength == 3 {
			nu[2]++
		} else if maxRunLength == 4 {
			nu[3]++
		} else if maxRunLength == 5 {
			nu[4]++
		} else {
			nu[5]++
		}
	}

	// Calculate the test statistic chiSquared
	chiSquared := 0.0
	piValues := []float64{0.2148, 0.3672, 0.2305, 0.1250, 0.0463, 0.0160}
	for i := 0; i < 6; i++ {
		chiSquared += math.Pow(float64(nu[i])-float64(numBlocks)*piValues[i], 2) / (float64(numBlocks) * piValues[i])
	}

	// Calculate the P-value
	pValue := math.Exp(-chiSquared / 2)
	alpha := 0.001

	return pValue, pValue > alpha
}

// SpectralTest performs the spectral test on the input sequence.
// The input sequence should be a slice of integers, where each integer
// represents the value of a bit (either 0 or 1).
// The function returns a boolean value indicating whether the input
// sequence passes the test (true) or fails the test (false).
func SpectralTest(bitArray *BitArray) (float64, bool) {
	n := bitArray.Size()

	// Prepare data for the Fourier transform
	data := make([]float64, n)
	for i := 0; i < n; i++ {
		if bitArray.Get(i) {
			data[i] = 1.0
		} else {
			data[i] = -1.0
		}
	}

	// Perform the Fourier transform
	fft := fourier.NewFFT(len(data))
	cdata := fft.Coefficients(nil, data)

	// Compute the magnitudes of the first n/2 complex coefficients
	magnitudes := make([]float64, n/2)
	for i := 0; i < n/2; i++ {
		magnitudes[i] = cmplx.Abs(cdata[i])
	}

	// Calculate the test statistic (T) and the p-value
	T := math.Sqrt(float64(n)-1.0) * math.Sqrt(2.0) / 3.0
	pValue := math.Erfc(T / (math.Sqrt(2.0) * math.Sqrt(float64(n)/4.0)))

	alpha := 0.001
	pass := pValue >= alpha
	return pValue, pass
}

// UniversalStatisticalTest performs Maurer's "universal statistical" test on the input sequence.
// The input sequence should be a slice of integers, where each integer
// represents the value of a bit (either 0 or 1).
// The function returns a boolean value indicating whether the input
// sequence passes the test (true) or fails the test (false).
func UniversalStatisticalTest(data *BitArray) (float64, bool) {
	n := data.Size()
	fmt.Println(n)
	if n < 1000 {
		return -1.0, false
	}

	var L, Q, K int
	if n >= 1010 && n < 3864 {
		L, Q, K = 3, 10, 64
	} else if n >= 3864 && n < 904964 {
		L, Q, K = 5, 10, 1024
	} else {
		L, Q, K = 6, 10, 4096
	}

	T := make([]int, n)
	for i := 1; i < n; i++ {
		if data.Get(i - 1) {
			T[i] = 1 + T[i-1]
		} else {
			T[i] = T[i-1]
		}
	}

	initTable := make([]int, K)
	for i := 0; i < K; i++ {
		initTable[i] = 0
	}
	for i := n - 1; i >= n-Q*(1<<L); i-- {
		index := (1 << L) - 1 - T[i]
		if index >= 0 && index < K {
			initTable[index]++
		}
	}

	sum := 0.0
	for i := 0; i < K; i++ {
		p := float64(initTable[i]) / float64(Q)
		if p > 0.0 {
			sum += p * math.Log2(p)
		}
	}
	phi := -sum

	c := 0.7 - 0.8/float64(L) + (4+32/float64(L))*math.Pow(float64(n), -3.0/float64(L))/15
	v := c * math.Sqrt(math.Pow(2.0, float64(L)-1)*float64(K)/float64(Q))
	delta := phi - c*float64(K) + v

	pValue := math.Erfc(math.Abs(delta) / (math.Sqrt(2.0) * v))
	alpha := 0.001

	return pValue, pValue > alpha
}

func ApproximateEntropyTest(data *BitArray, blockSize int) (float64, bool) {
	n := data.Size()
	if n < blockSize {
		return -1.0, false
	}
	m := 10 // Block length (m = 10 for n >= 1000)
	r := 0.5

	// Function to get the m-bit block as a string
	getBlockAsString := func(data *BitArray, start, length int) string {
		block := ""
		for i := 0; i < length; i++ {
			if data.Get(start + i) {
				block += "1"
			} else {
				block += "0"
			}
		}
		return block
	}

	// Count the occurrences of each m-bit block
	counts := make(map[string]int)
	for i := 0; i < n-m+1; i++ {
		block := getBlockAsString(data, i, m)
		counts[block]++
	}

	// Calculate the sum of the frequencies
	sum := 0.0
	for _, count := range counts {
		sum += float64(count) * math.Log(float64(count)/float64(n-m+1))
	}

	// Calculate the approximate entropy (phi_m)
	phiM := sum / float64(n-m+1)

	// Count the occurrences of each (m+1)-bit block
	countsMPlus1 := make(map[string]int)
	for i := 0; i < n-m; i++ {
		block := getBlockAsString(data, i, m+1)
		countsMPlus1[block]++
	}

	// Calculate the sum of the frequencies for (m+1)-bit blocks
	sumMPlus1 := 0.0
	for _, count := range countsMPlus1 {
		sumMPlus1 += float64(count) * math.Log(float64(count)/float64(n-m))
	}

	// Calculate the approximate entropy (phi_m+1)
	phiMPlus1 := sumMPlus1 / float64(n-m)

	// Calculate the test statistic
	apEn := phiM - phiMPlus1

	// Calculate the P-value
	pValue := math.Erfc(math.Abs(apEn) / (math.Sqrt(2.0) * r * math.Sqrt(float64(n-m+1))))

	alpha := 0.001

	return pValue, pValue > alpha
}

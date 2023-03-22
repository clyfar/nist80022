package nist80022

import (
	"fmt"
	"math"
	"strconv"
	"strings"

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

type BitArray struct {
	data []byte
}

func NewBitArray(size int) *BitArray {
	return &BitArray{
		data: make([]byte, (size+7)>>3),
	}
}

func (b *BitArray) Set(index int, value bool) {
	if value {
		b.data[index>>3] |= 1 << (index & 7)
	} else {
		b.data[index>>3] &^= 1 << (index & 7)
	}
}

func (b *BitArray) Get(index int) bool {
	return (b.data[index>>3] & (1 << (index & 7))) != 0
}

func (b *BitArray) Size() int {
	return len(b.data) << 3
}

func IntArrayToBitArray(intArray []int) *BitArray {
	n := len(intArray) * 8
	bitArray := NewBitArray(n)

	for i, value := range intArray {
		for j := 0; j < 8; j++ {
			bit := (value >> (7 - j)) & 1
			bitArray.Set(i*8+j, bit == 1)
		}
	}

	return bitArray
}

// BinaryMatrixRankTest performs the binary matrix rank test on the input
// bit string. The test divides the bit string into m x q binary matrices,
// and calculates the rank of each matrix. It then calculates the chi-squared
// statistic from the deviation of the observed rank sum from the expected rank
// sum, and computes the corresponding p-value from the gamma distribution.
// The function returns the negative base 10 logarithm of the p-value as the score.
//
// bits: the input bit string to test.
// m: the number of rows in each matrix.
// q: the number of columns in each matrix.
func BinaryMatrixRankTest(bits string, m, q int) (float64, bool) {
	n := len(bits)
	numRows := n / m
	numMatrices := n / (m * q)
	var rankSum float64
	for i := 0; i < numMatrices; i++ {
		matrixStart := i * m * q
		matrixEnd := matrixStart + m*q
		matrixBits := bits[matrixStart:matrixEnd]
		matrix := make([][]int, m)
		for j := 0; j < m; j++ {
			matrix[j] = make([]int, q)
			for k := 0; k < q; k++ {
				matrix[j][k] = int(matrixBits[j*q+k] - '0')
			}
		}
		rank := calculateMatrixRank(matrix, m, q)
		rankSum += float64(rank)
	}
	expectedRank := float64(q*(q+1)/2) / (math.Pow(2, float64(q)))
	expectedSum := float64(numRows) * expectedRank
	chiSq := math.Pow(rankSum-expectedSum, 2) / expectedSum
	gammaDist := distuv.Gamma{Alpha: float64(numRows) / 2, Beta: 0.5}
	pval := 1 - gammaDist.CDF(chiSq/2)
	score := -math.Log10(pval)
	return score, score > 5
}

// calculateMatrixRank computes the rank of a binary matrix using Gaussian
// elimination with partial pivoting.
//
// matrix: the binary matrix to calculate the rank of.
// m: the number of rows in the matrix.
// q: the number of columns in the matrix.
func calculateMatrixRank(matrix [][]int, m, q int) int {
	rank := 0
	for j := 0; j < q; j++ {
		// Find the row with the largest absolute value in column j
		maxRowIndex := j
		for i := j + 1; i < m; i++ {
			if math.Abs(float64(matrix[i][j])) > math.Abs(float64(matrix[maxRowIndex][j])) {
				maxRowIndex = i
			}
		}

		// Swap rows to put the maximum element at position (j,j)
		if maxRowIndex != j {
			matrix[j], matrix[maxRowIndex] = matrix[maxRowIndex], matrix[j]
		}

		// Check if the element at position (j,j) is non-zero
		if matrix[j][j] == 0 {
			continue
		}

		// Zero out all elements in column j below the diagonal element
		for i := j + 1; i < m; i++ {
			if matrix[i][j] != 0 {
				for k := j + 1; k < q; k++ {
					matrix[i][k] = (matrix[i][k] + matrix[j][k]) % 2
				}
			}
		}

		rank++
	}
	return rank
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
func MonobitTest(bits string) (float64, bool) {
	n := len(bits)
	ones := countOnes(bits)
	expectedOnes := float64(n) / 2.0
	sigma := math.Sqrt(float64(n) / 4.0)
	z := (float64(ones) - expectedOnes) / sigma
	pval := 1 - math.Erf(math.Abs(z)/math.Sqrt2)
	pass := pval >= 0.01
	return pval, pass
}

// BlockFrequencyTest performs the block frequency test on the input sequence.
// The input sequence should be a slice of integers, where each integer
// represents the value of a bit (either 0 or 1)
// The function returns a boolean value indicating whether the input
// sequence passes the test (true) or fails the test (false).
func BlockFrequencyTest(bits string, blocksize int) (float64, bool) {
	n := len(bits)
	numBlocks := n / blocksize

	// Calculate the expected proportion of ones in each block
	expectedOnes := float64(countOnes(bits)) / float64(n)
	expectedZeros := 1.0 - expectedOnes

	// Calculate the test statistic for each block
	chisq := 0.0
	for i := 0; i < numBlocks; i++ {
		blockStart := i * blocksize
		blockEnd := (i + 1) * blocksize
		blockBits := bits[blockStart:blockEnd]
		ones := countOnes(blockBits)
		zeros := blocksize - ones
		proportions := []float64{float64(ones) / float64(blocksize), float64(zeros) / float64(blocksize)}
		expectedCounts := []float64{float64(blocksize) * expectedOnes, float64(blocksize) * expectedZeros}
		chisq += chiSquared(proportions, expectedCounts)
	}

	// Calculate the p-value and pass/fail
	pval := 1 - chiSquaredCDF(chisq, numBlocks*2-2)
	pass := pval >= 0.01
	return pval, pass
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
func SpectralTest(bits string) (float64, bool) {
	n := len(bits)
	m := int(math.Floor(float64(n) / 2.0))
	if m <= 0 {
		return 0.0, false
	}

	var s []float64 = make([]float64, m)
	for i := 0; i < m; i++ {
		s[i] = 0.0
		for j := 0; j < n-i; j++ {
			bit1, _ := strconv.Atoi(string(bits[j]))
			bit2, _ := strconv.Atoi(string(bits[j+i]))
			s[i] += float64((2*bit1 - 1) * (2*bit2 - 1))
		}
	}

	var tau float64
	tau = 0.0
	for i := 1; i < m; i++ {
		tau += s[i] / s[0]
	}
	tau = 2.0 * tau

	var pval float64 = math.Erfc(math.Abs(tau) / math.Sqrt(2.0))

	return pval, pval > 0.01
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

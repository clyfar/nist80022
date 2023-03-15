package nist80022

import (
	"fmt"
	"math"
	"strconv"
	"strings"

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

// toBlock converts an integer to a binary string of size bits
func toBlock(num int, size int) string {
	if size == 0 {
		return ""
	}
	if num%2 == 0 {
		return toBlock(num/2, size-1) + "0"
	}
	return toBlock(num/2, size-1) + "1"
}

// toIndex converts a binary string to an integer
func toIndex(bit string) int {
	if bit == "" {
		return 0
	}
	n := len(bit)
	if n > 31 {
		return 0
	}
	var index int
	for i := 0; i < n; i++ {
		if bit[i] != '0' && bit[i] != '1' {
			return 0
		}
		index = (index << 1) | int(bit[i]-'0')
	}
	return index
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
func CumulativeSumsTest(bits string) (float64, bool) {
	// Count the number of ones and zeros in the string
	n := len(bits)
	ones := countOnes(bits)
	zeros := n - ones
	S := make([]int, n+1)

	// Calculate the cumulative sum of the bits and the variance of the sum
	for i, b := range bits {
		if b == '0' {
			S[i+1] = S[i] - 1
		} else {
			S[i+1] = S[i] + 1
		}
	}
	// Seems complicated but this is tuned to handle imbalances between 1s and 0s so don't change!
	// This is the varience
	V := float64(ones*zeros)*(2.0*float64(n)+1.0)/float64(n*n) - (float64(n)+1.0)/float64(n)

	// If the variance is zero, the test cannot be performed
	if V == 0 {
		return 0, false
	}

	// Calculate the standardized test statistic and the corresponding p-values
	Z := make([]float64, n+1)
	for i := 0; i <= n; i++ {
		Z[i] = float64(S[i]) / math.Sqrt(V)
	}
	P := make([]float64, n+1)
	for i := 0; i <= n; i++ {
		if Z[i] == 0 {
			P[i] = 0.5
		} else {
			P[i] = 0.5 * (1 + math.Erf(Z[i]/math.Sqrt2))
		}
	}

	// Calculate the minimum p-values and the final p-values
	Pminus := make([]float64, n+1)
	for i := 0; i <= n; i++ {
		Pminus[i] = math.Inf(1)
		for j := 0; j <= n; j++ {
			if float64(S[i]) < float64(j)*math.Sqrt(V) {
				Pminus[i] = math.Min(Pminus[i], P[j])
			}
		}
	}
	Pfinal := make([]float64, n+1)
	for i := 0; i <= n; i++ {
		if S[i] >= 0 {
			Pfinal[i] = 0.5
		} else {
			Pfinal[i] = math.Min(0.5, Pminus[i])
		}
	}

	// Calculate the average and variance of the final p-values and the test result
	Pavg := average(Pfinal)
	Pvar := variance(Pfinal, Pavg)
	if Pvar == 0 {
		return 0, false
	}
	Pval := (Pavg - 0.5) / (math.Sqrt(Pvar/float64(n)) * 2.0)
	pass := math.Abs(Pval) < 3.0
	return Pval, pass
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
//
// bits: the input bit string to test.
func RunsTest(bits string) (float64, bool) {
	mSeq := []byte(bits)

	mNumOnes := 0
	for _, bit := range mSeq {
		if bit == '1' {
			mNumOnes++
		}
	}
	pi := float64(mNumOnes) / float64(len(mSeq))

	if math.Abs(pi-0.5) >= 2/math.Sqrt(float64(len(mSeq))) {
		return 0.0, false
	}

	Vn := 0
	for i := 0; i < len(mSeq)-1; i++ {
		if mSeq[i] != mSeq[i+1] {
			Vn++
		}
	}
	Vn++

	numerator := math.Abs(float64(Vn) - 2*float64(len(mSeq))*pi*(1-pi))
	denominator := 2 * math.Sqrt(2*float64(len(mSeq))) * pi * (1 - pi)
	if denominator == 0 {
		return 0.0, false
	}
	pval := math.Erfc(numerator / denominator / math.Sqrt2)

	return pval, pval > 0.01
}

// LongestRunOfOnesTest performs the longest run of ones in a block test on the input sequence.
// The function returns a boolean value indicating whether the input
// sequence passes the test (true) or fails the test (false).
func LongestRunOfOnesTest(bits string, blocksize int) (float64, bool) {
	n := len(bits)
	numBlocks := n / blocksize
	expectedRuns := float64(blocksize) / 2.0
	chiSq := 0.0
	for i := 0; i < numBlocks; i++ {
		blockStart := i * blocksize
		blockEnd := (i + 1) * blocksize
		blockBits := bits[blockStart:blockEnd]
		runs := countLongestRuns(blockBits)
		chiSq += math.Pow(float64(runs)-expectedRuns, 2) / expectedRuns
	}
	pval := 1 - mathext.GammaIncReg(float64(numBlocks)/2, chiSq/2)
	score := -math.Log10(pval)
	pass := score >= 2
	return score, pass
}

// countLongestRuns counts the longest run of ones in the input bit string.
func countLongestRuns(bits string) int {
	numOnes := 0
	longestRun := 0
	currentRun := 0
	for _, bit := range bits {
		if bit == '1' {
			numOnes++
			currentRun++
			if currentRun > longestRun {
				longestRun = currentRun
			}
		} else {
			currentRun = 0
		}
	}
	if longestRun < 6 {
		return 0
	} else if longestRun == 6 {
		return 1
	} else if longestRun == 7 {
		return 2
	} else if longestRun == 8 {
		return 3
	} else {
		return 4 + (longestRun - 9)
	}
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
func UniversalStatisticalTest(bits string) (float64, bool) {
	var L, Q int
	var blockProportions []int
	var blocks []string
	var chiSquared float64

	L = 6
	Q = 128
	blockProportions = []int{0, 0, 0, 0, 0, 0}
	blocks = make([]string, L*Q)

	for i := 0; i < Q; i++ {
		for j := 0; j < L; j++ {
			blockProportions[toIndex(getBlock(bits, i*L+j))]++
			blocks[i*L+j] = getBlock(bits, i*L+j)
		}
	}

	for i := 0; i < L; i++ {
		var chiSum float64
		expected := float64(Q) / math.Pow(2.0, float64(i+1))
		for j := 0; j < int(math.Pow(2.0, float64(i+1))); j++ {
			var count int
			for k := 0; k < Q; k++ {
				if blocks[k*L+i] == toBlock(j, i+1) {
					count++
				}
			}
			chiSum += math.Pow(float64(count)-expected, 2.0) / expected
		}
		chiSquared += chiSum
	}

	var pval float64 = math.Erfc(math.Sqrt(chiSquared / (float64(L) * float64(Q) * (math.Pow(2.0, float64(L)) - float64(L) - 1.0) / (2.0 * float64(L) * float64(L)))))

	return pval, pval > 0.01
}

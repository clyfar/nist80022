package nist80022

import (
	"math"
	"reflect"
	"testing"
)

func TestCountOnes(t *testing.T) {
	type test struct {
		name  string
		input string
		want  int
	}

	tests := []test{
		{
			name:  "Empty input",
			input: "",
			want:  0,
		},
		{
			name:  "All 0s input",
			input: "00000000",
			want:  0,
		},
		{
			name:  "Has one 1 input",
			input: "00000001",
			want:  1,
		},
		{
			name:  "two 1s input",
			input: "00000011",
			want:  2,
		},
		{
			name:  "Changes order order of 0s and 1s input",
			input: "01100011",
			want:  4,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := countOnes(tc.input)
			if got != tc.want {
				t.Errorf("countOnes(%s) = %d, want %d", tc.input, got, tc.want)
			}
		})
	}
}

func TestAverage(t *testing.T) {
	type test struct {
		name  string
		input []float64
		want  float64
	}

	tests := []test{
		{
			name:  "Empty input",
			input: []float64{},
			want:  0,
		},
		{
			name:  "One number input",
			input: []float64{1},
			want:  1,
		},
		{
			name:  "Three number input",
			input: []float64{1, 2, 3},
			want:  2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := average(tc.input)
			if got != tc.want {
				t.Errorf("average(%v) = %f, want %f", tc.input, got, tc.want)
			}
		})
	}
}

func TestVariance(t *testing.T) {
	type test struct {
		name string
		nums []float64
		avg  float64
		want float64
	}

	tests := []test{
		{
			name: "Empty input",
			nums: []float64{},
			avg:  0.0,
			want: 0.0,
		},
		{
			name: "One number input",
			nums: []float64{1.0},
			avg:  1.0,
			want: 0.0,
		},
		{
			name: "Two number input",
			nums: []float64{1.0, 2.0},
			avg:  1.5,
			want: 0.25,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := variance(tc.nums, tc.avg)
			if got != tc.want {
				t.Errorf("variance(%v) = %f, want %f", tc.nums, got, tc.want)
			}
		})
	}
}

func TestIntsToBits(t *testing.T) {
	type test struct {
		name  string
		input []int
		want  string
	}

	tests := []test{
		{
			name:  "Empty input",
			input: []int{},
			want:  "",
		},
		{
			name:  "Two number input",
			input: []int{1, 2},
			want:  "110",
		},
		{
			name:  "Many number input",
			input: []int{1, 2, 3, 4, 5, 6, 7, 8, 1111, 2222, 3333456},
			want:  "110111001011101111000100010101111000101011101100101101110101010000",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := IntsToBits(tc.input)
			if got != tc.want {
				t.Errorf("intsToBits(%v) = %s, want %s", tc.input, got, tc.want)
			}
		})
	}
}

func TestChiSquared(t *testing.T) {
	type test struct {
		name           string
		counts         []float64
		expectedCounts []float64
		want           float64
	}

	tests := []test{
		{
			name:           "Empty input",
			counts:         []float64{},
			expectedCounts: []float64{},
			want:           0.0,
		},
		{
			name:           "One number input",
			counts:         []float64{1.0},
			expectedCounts: []float64{2.0},
			want:           0.5,
		},
		{
			name:           "Three number input",
			counts:         []float64{1.0, 2.0, 3.0},
			expectedCounts: []float64{5.1, 3.2, 4.1},
			want:           4.041200382592061,
		},
		{
			name:           "Same number input",
			counts:         []float64{1.0, 2.0, 3.0},
			expectedCounts: []float64{1.0, 2.0, 3.0},
			want:           0.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := chiSquared(tc.counts, tc.expectedCounts)
			if got != tc.want {
				t.Errorf("chiSquared(%v, %v) = %f, want %f", tc.counts, tc.expectedCounts, got, tc.want)
			}
		})
	}
}

func TestChiSquaredCDF(t *testing.T) {
	type test struct {
		name  string
		chisq float64
		df    int
		want  float64
	}

	tests := []test{
		{
			name:  "Determine the p-val between 0 and 1",
			chisq: 0.0,
			df:    1,
			want:  1.0,
		},
		{
			name:  "Determine the p-val between 2.5 and 3",
			chisq: 2.5,
			df:    3,
			want:  0.475291,
		},
		{
			name:  "Determine the p-val between 5.8 and 10",
			chisq: 5.8,
			df:    10,
			want:  0.831777,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := chiSquaredCDF(tc.chisq, tc.df)
			if math.Abs(got-tc.want) > 0.0001 {
				t.Errorf("chiSquaredCDF(%f, %d) = %f, want %f", tc.chisq, tc.df, got, tc.want)
			}
		})
	}
}

func TestBinaryStringToIntSlice(t *testing.T) {
	type test struct {
		name  string
		input string
		want  []int
	}

	tests := []test{
		{
			name:  "Empty input",
			input: "",
			want:  []int{},
		},
		{
			name:  "single digit input",
			input: "1",
			want:  []int{1},
		},
		{
			name:  "multiple digit input",
			input: "11011000001",
			want:  []int{1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1},
		},
		{
			name:  "invalid input",
			input: "123",
			want:  []int{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := BinaryStringToIntSlice(tc.input)
			if got != nil && !reflect.DeepEqual(got, tc.want) {
				t.Errorf("binaryStringToIntSlice(%s) = %v, want %v", tc.input, got, tc.want)
			}
			if err != nil && got != nil {
				t.Errorf("binaryStringToIntSlice(%s) = %v, want %v", tc.input, err, nil)
			}
		})
	}
}

func TestGetBlock(t *testing.T) {
	type test struct {
		name       string
		str        string
		startIndex int
		want       string
	}

	tests := []test{
		{
			name:       "Empty input",
			str:        "",
			startIndex: 0,
			want:       "",
		},
		{
			name:       "Single character input",
			str:        "a",
			startIndex: 0,
			want:       "a",
		},
		{
			name:       "Multiple character input",
			str:        "abc",
			startIndex: 0,
			want:       "abc",
		},
		{
			name:       "Multiple character input with non-zero start index",
			str:        "abc",
			startIndex: 3,
			want:       "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := getBlock(tc.str, tc.startIndex)
			if got != tc.want {
				t.Errorf("getBlock(%s, %d) = %s, want %s", tc.str, tc.startIndex, got, tc.want)
			}
		})
	}
}

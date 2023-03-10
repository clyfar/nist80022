package nist80022

import (
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
			name:  "All zeros",
			input: "00000000",
			want:  0,
		},
		{
			name:  "Has one",
			input: "00000001",
			want:  1,
		},
		{
			name:  "Has two",
			input: "00000011",
			want:  2,
		},
		{
			name:  "Change order",
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
			name:  "Empty",
			input: []float64{},
			want:  0,
		},
		{
			name:  "One",
			input: []float64{1},
			want:  1,
		},
		{
			name:  "Two",
			input: []float64{1, 2},
			want:  1.5,
		},
		{
			name:  "Three",
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
			name: "Empty",
			nums: []float64{},
			avg:  0.0,
			want: 0.0,
		},
		{
			name: "One",
			nums: []float64{1.0},
			avg:  1.0,
			want: 0.0,
		},
		{
			name: "Two",
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

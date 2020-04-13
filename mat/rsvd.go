// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"math/rand"
)

// RSVD is a type for creating and using the Randomized Singular Value Decomposition (RSVD)
// of a matrix.
type RSVD struct {
	svd  *SVD
	rank int
	q    *Dense
	m    int
}

// Factorize computes the randomized singular value decomposition (RSVD) of the input matrix A
// using randomized matrix rank × rank
//
// Factorize returns whether the decomposition succeeded. If the decomposition
// failed, routines that require a successful factorization will panic.
// Factorize will also panic if rank is too low
func (rsvd *RSVD) Factorize(A Matrix, rank int) bool {

	const minRank = 1

	// Check if rank is too small
	if rank < minRank {
		panic(fmt.Sprintf("Rank %d must be at least %d", rank, minRank))
	}

	// Dimensions of input matrix:
	// [A] = m × n
	m, n := A.Dims()

	// Create random matrix:
	// [P] = n × rank
	P := makeRandomMatrix(n, rank)

	// Project random matrix P into original M:
	// [Z] = [M × P] = (m × n) × (n × rank) = m × rank
	Z := NewDense(m, rank, nil)
	Z.Mul(A, P)

	// Factorize M into orthogonal Q and triangular R:
	// [QFull] = m × m
	var QFull Dense
	QR := QR{}
	QR.Factorize(Z)
	QR.QTo(&QFull)

	// Truncate QFull:
	// [Q] = m × rank
	Q := QFull.Slice(0, m, 0, rank).(*Dense)

	// Project M into Q:
	// [Y] = [Qᵀ × M] = (rank × m) × (m × n) = rank × n
	Y := NewDense(rank, n, nil)
	Y.Mul(Q.T(), A)

	rsvd.m = m
	rsvd.q = Q

	// Perform SVD for Y:
	// [Y] = [Uy × Σ × V] = (rank × rank) × (rank × rank) × (rank × n) = rank × n
	return rsvd.svd.Factorize(Y, SVDThin)
}

// Values returns the singular values of the factorized matrix in descending order.
//
// If the input slice is non-nil, the values will be stored in-place into
// the slice. In this case, the slice must have length min(m,n), and Values will
// panic with ErrSliceLengthMismatch otherwise. If the input slice is nil, a new
// slice of the appropriate length will be allocated and returned.
//
// Values will panic if the receiver does not contain a successful factorization.
func (rsvd *RSVD) Values(s []float64) []float64 {
	return rsvd.svd.Values(s)
}

// UTo extracts the matrix U from the singular value decomposition..
//
// If dst is empty, UTo will resize dst to be m×rank. When dst is non-empty, then
// UTo will panic if dst is not the appropriate size. UTo will also panic if
// the receiver does not contain a successful factorization, or if U was
// not computed during factorization.
func (rsvd *RSVD) UTo(dst *Dense) {
	var Uy Dense
	// Uy := NewDense(rsvd.rank, rsvd.rank, nil)
	rsvd.svd.UTo(&Uy)

	// Project Uy into QS:
	// [U] = [QS × Uy] = (m × rank) × (rank × rank) = m × rank
	U := NewDense(rsvd.m, rsvd.rank, nil)
	U.Mul(rsvd.q, &Uy)

	if dst.IsEmpty() {
		dst.ReuseAs(rsvd.m, rsvd.rank)
	} else {
		r2, c2 := dst.Dims()
		if rsvd.m != r2 || rsvd.rank != c2 {
			panic(ErrShape)
		}
	}

	dst.Copy(U)
}

// VTo extracts the matrix V from the randomized singular value decomposition
//
// If dst is empty, VTo will resize dst to be n×rank. When dst is non-empty, then
// VTo will panic if dst is not the appropriate size. VTo will also panic if
// the receiver does not contain a successful factorization, or if V was
// not computed during factorization.
func (rsvd *RSVD) VTo(dst *Dense) {
	rsvd.svd.VTo(dst)
}

// makeRandomMatrix creates random matrix with given amount of rows and cols
func makeRandomMatrix(rows, columns int) *Dense {
	dataLength := rows * columns
	data := make([]float64, dataLength, dataLength)

	for i := range data {
		data[i] = rand.Float64()
	}

	return NewDense(rows, columns, data)
}

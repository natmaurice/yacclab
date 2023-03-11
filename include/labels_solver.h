// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELS_SOLVER_H_
#define YACCLAB_LABELS_SOLVER_H_

#include "register.h"
#include "memory_tester.h"
#include "performance_evaluator.h"
#include <lsl3dlib/features.hpp>

#include <cassert>


// Union-find (UF)
class UF {
    // Maximum number of labels (included background) = 2^(sizeof(unsigned) x 8)
public:
    UF() : P_(nullptr), length_(0), start_index(0), end_index(0), mem_owner_(true) {
    }

    UF(unsigned* P, unsigned max_length) : P_(P), length_(0), start_index(0), end_index(0), mem_owner_(false) {
    }
    
     double Alloc(unsigned max_length, PerformanceEvaluator& perf) {
	 Dealloc();
        perf.start();
        P_ = new unsigned[max_length];
        memset(P_, 0, max_length * sizeof(unsigned));
        perf.stop();
        double t = perf.last();
        perf.start();
        memset(P_, 0, max_length * sizeof(unsigned));
        perf.stop();
        return t - perf.last();
    }
     void Alloc(unsigned max_length) {
	Dealloc();
        P_ = new unsigned[max_length];
    }
     void Dealloc() {
	 if (mem_owner_) {
	     delete[] P_;
	 }
	 P_ = nullptr;
	mem_owner_ = true;
    }
    
     void Setup() {
        P_[0] = 0;// First label is for background pixels
        length_ = 1;
	start_index = 1;
	end_index = 1;
    }
     unsigned NewLabel() {
	 int label = end_index;
	 P_[end_index] = label;
	 end_index++;
	 return label;
    }
     unsigned GetLabel(unsigned index) {
	 //assert(index < end_index);
        return P_[index];
    }

    // Basic functions of the UF solver required only by the Light-Speed Labeling Algorithms:
    // - "UpdateTable" updates equivalences array without performing "Find" operations
    // - "FindRoot" finds the root of the tree of node i (for the other algorithms it is 
    //		already included in the "Merge" and "Flatten" functions).
    void UpdateTable(unsigned e, unsigned r) {
        assert(e >= r);
	P_[e] = r;
    }
     unsigned FindRoot(unsigned root) {
        while (P_[root] < root) {
            root = P_[root];
        }
        return root;
    }

    void SubTable(UF& subET, unsigned start) {
	subET.P_ = P_;	
	subET.length_ = start;
	subET.start_index = start;
	subET.end_index = start;
    }
    
     unsigned Merge(unsigned i, unsigned j)
    {
        // FindRoot(i)
        while (P_[i] < i) {
            i = P_[i];
        }

        // FindRoot(j)
        while (P_[j] < j) {
            j = P_[j];
        }

        if (i < j)
            return P_[j] = i;
        return P_[i] = j;
    }


    void FlattenNoCompress() {
	for (unsigned i = start_index; i < end_index; ++i) {
	    if (P_[i] < i) {
		P_[i] = P_[P_[i]];
	    }
	}
    }
    
    template <typename ConfFeatures>
    void FlattenNoCompressFeatures(Features& features) {
	for (unsigned i = start_index; i < end_index; ++i) {
	    unsigned par = P_[i];
	    if (par < i) {
		unsigned ppar = P_[par];
		P_[i] = ppar;
		if (par != ppar) {
		    //std::cout << "Merging " << i << " with " << par << "\n";
		    features.Merge<ConfFeatures>(i, par);
		}
	    }
	}
    }

    
    unsigned Flatten()
    {
        unsigned k = start_index;
        for (unsigned i = start_index; i < end_index; ++i) {
            if (P_[i] < i) {
                P_[i] = P_[P_[i]];
		//std::cout << "1) ET[" << i << "] = " << P_[P_[i]] << "\n";
            }
            else {
                P_[i] = k;
		//std::cout << "2) ET[" << i << "] = " << k << "\n";
                k = k + 1;
            }
        }
	
	if (end_index > start_index) {
	    //end_index = k;
	}
	return k - start_index;
    }

    
    template <typename ConfFeatures>
    unsigned Flatten(Features& features)
    {
        unsigned k = start_index;
        for (unsigned i = start_index; i < end_index; ++i) {
            if (P_[i] < i) {

		//if (P_[P_[i]] == 42) {
		    //std::cout << "1. (" << i <<  ") P_[" << P_[i] << "] = " << P_[P_[i]] << "\n";
		//}
                P_[i] = P_[P_[i]];		
            }
            else {
                P_[i] = k;
		//std::cout << "2. P[" << i << "] = " << k << "\n";
		features.Shift<ConfFeatures>(k, i);
                k = k + 1;
            }
        }
	
	if (end_index > start_index) {
	    //end_index = k;
	}	
        return k - start_index;
    }

    template <typename ConfFeatures>
    unsigned FlattenWithFeatures(Features& features) {
        unsigned k = start_index;
        for (unsigned i = start_index; i < end_index; ++i) {
	    unsigned label = P_[i];
            if (label < i) {
		unsigned par = P_[label];
		features.Merge<ConfFeatures>(i, par);
                P_[i] = par;
            }
            else {
		features.Shift<ConfFeatures>(k, i);
                P_[i] = k;
                k = k + 1;
            }
        }	
	
	if (end_index > start_index) {
	    //end_index = k;
	}	
	return k - start_index;
    }
    
    // Flatten a disjoint set of LabelsSolver into a single one
    // This assumes that labels solvers share same memory pool
    static unsigned MultiFlatten(UF* ET_chunks, size_t chunk_count) {
	unsigned k = 0;

	assert(chunk_count > 0);
	UF* ET = &ET_chunks[0];
	unsigned* __restrict__ P_ = ET->P_;

	unsigned start = ET->start_index;
	k = start;
	for (size_t chunk = 0; chunk < chunk_count; chunk++) {
	    UF* ET = &ET_chunks[chunk];
	    unsigned start_index = ET->start_index;
	    unsigned end_index = ET->end_index;
	    
	    for (unsigned i = start_index; i < end_index; i++) {
		if (P_[i] < i) {
		    P_[i] = P_[P_[i]];
		} else {
		    P_[i] = k;
		    k++;
		}		
	    }
	}
	return k - start;
    }

    // Flatten a disjoint set of LabelsSolver into a single one
    // This assumes that labels solvers share same memory pool
    template <typename ConfFeatures>
    static unsigned MultiFlatten(UF* ET_chunks, size_t chunk_count, Features& features) {
	unsigned k = 0;

	assert(chunk_count > 0);
	UF* ET = &ET_chunks[0];
	unsigned* __restrict__ P_ = ET->P_;

	unsigned start = ET->start_index;
	k = start;
	for (size_t chunk = 0; chunk < chunk_count; chunk++) {
	    UF* ET = &ET_chunks[chunk];
	    unsigned start_index = ET->start_index;
	    unsigned end_index = ET->end_index;
	    
	    for (unsigned i = start_index; i < end_index; i++) {
		if (P_[i] < i) {
		    P_[i] = P_[P_[i]];
		} else {
		    P_[i] = k;
		    features.Shift<ConfFeatures>(k, i);
		    k++;
		}		
	    }
	}
	return k - start;
    }

    
    unsigned long Size() const {
	return end_index - start_index;
    }
    
    /***************************************************************/

     void MemAlloc(unsigned max_length)
    {
        mem_P_ = MemVector<unsigned>(max_length);
    }
     void MemDealloc() {}
     void MemSetup() {
        mem_P_[0] = 0;	 // First label is for background pixels
        length_ = 1;
    }
     unsigned MemNewLabel() {
        mem_P_[length_] = length_;
        return length_++;
    }
     unsigned MemGetLabel(unsigned index) {
        return mem_P_[index];
    }

     double MemTotalAccesses() {
        return mem_P_.GetTotalAccesses();
    }

    // Basic functions of the UF solver required only by the Light-Speed Labeling Algorithms:
    // - "MemUpdateTable" updates equivalences array without performing "MemFind" operations
    // - "MemFindRoot" finds the root of the tree of node i (for the other algorithms it is 
    //		already included in the "MemMerge" and "MemFlatten" functions).
     void MemUpdateTable(unsigned e, unsigned r) {
        mem_P_[e] = r;
    }
     unsigned MemFindRoot(unsigned root) {
        while (mem_P_[root] < root) {
            root = mem_P_[root];
        }
        return root;
    }

     unsigned MemMerge(unsigned i, unsigned j)
    {
        // FindRoot(i)
        while (mem_P_[i] < i) {
            i = mem_P_[i];
        }

        // FindRoot(j)
        while (mem_P_[j] < j) {
            j = mem_P_[j];
        }

        if (i < j)
            return mem_P_[j] = i;
        return mem_P_[i] = j;
    }
     unsigned MemFlatten()
    {
        unsigned k = 1;
        for (unsigned i = 1; i < length_; ++i) {
            if (mem_P_[i] < i) {
                mem_P_[i] = mem_P_[mem_P_[i]];
            }
            else {
                mem_P_[i] = k;
                k = k + 1;
            }
        }
        return k;
    }
public:
    unsigned *P_;
    unsigned length_;
    unsigned start_index, end_index;
    bool mem_owner_;
private:
     MemVector<unsigned> mem_P_;
};

// Union-Find (UF) with path compression (PC) as in:
// Two Strategies to Speed up Connected Component Labeling Algorithms
// Kesheng Wu, Ekow Otoo, Kenji Suzuki
class UFPC {
    // Maximum number of labels (included background) = 2^(sizeof(unsigned) x 8)
public:
    UFPC() : P_(nullptr), length_(0), mem_owner_(true) {
    }

    UFPC(unsigned* P, unsigned max_length) : P_(P), mem_owner_(false) {
    }
    
     double Alloc(unsigned max_length, PerformanceEvaluator& perf) {
	 Dealloc();
        perf.start();
        P_ = new unsigned[max_length];
        memset(P_, 0, max_length * sizeof(unsigned));
        perf.stop();
        double t = perf.last();
        perf.start();
        memset(P_, 0, max_length * sizeof(unsigned));
        perf.stop();
        return t - perf.last();
    }
    
    void Alloc(unsigned max_length) {
	Dealloc();
        P_ = new unsigned[max_length];
    }
    
    void Dealloc() {
	if (mem_owner_) {
	    delete[] P_;
	}
	P_ = nullptr;
	mem_owner_ = true;
    }
    
    void Setup() {
        P_[0] = 0;	 // First label is for background pixels
        length_ = 1;
	start_index = 1;
	end_index = 1;
    }

    void SubTable(UFPC& subET, unsigned start_index) {
	subET.P_ = P_;
	subET.length_ = 0;
	subET.start_index = start_index;
	subET.end_index = start_index;
    }
    
    unsigned NewLabel() {
	int label = end_index;
        P_[end_index] = label;
	end_index++;
        return label;
    }
    
    unsigned GetLabel(unsigned index) {
        return P_[index];
    }

    unsigned FindRoot(unsigned i) {
	unsigned root = i;
	while (P_[root] < root) {
	    root = P_[root];
	}
	return root;
    }

    void UpdateTable(unsigned e, unsigned r) {
	assert(e >= r);
	P_[e] = r;
    }   
    
     unsigned Merge(unsigned i, unsigned j)
    {
        // FindRoot(i)
        unsigned root(i);
        while (P_[root] < root) {
            root = P_[root];
        }
        if (i != j) {
            // FindRoot(j)
            unsigned root_j(j);
            while (P_[root_j] < root_j) {
                root_j = P_[root_j];
            }
            if (root > root_j) {
                root = root_j;
            }
            // SetRoot(j, root);
            while (P_[j] < j) {
                unsigned t = P_[j];
                P_[j] = root;
                j = t;
            }
            P_[j] = root;
        }
        // SetRoot(i, root);
        while (P_[i] < i) {
            unsigned t = P_[i];
            P_[i] = root;
            i = t;
        }
        P_[i] = root;
        return root;
    }


    void FlattenNoCompress() {
	for (unsigned i = start_index; i < end_index; ++i) {
	    if (P_[i] < i) {
                P_[i] = P_[P_[i]];
            }
	}
    }

    template <typename ConfFeatures>
    void FlattenNoCompressFeatures(Features& features) {
	for (unsigned i = start_index; i < end_index; ++i) {
	    unsigned par = P_[i];
	    if (par < i) {
		P_[i] = P_[par];
		features.Merge<ConfFeatures>(i, par);
	    }
	}
    }

    
    unsigned Flatten()
    {
        unsigned k = start_index;
        for (unsigned i = start_index; i < end_index; ++i) {
            if (P_[i] < i) {
                P_[i] = P_[P_[i]];
            }
            else {
                P_[i] = k;
                k = k + 1;
            }
        }

	if (end_index > start_index) {
	    end_index = k;
	}
        return k;
    }

    
    template <typename ConfFeatures>
    unsigned Flatten(Features& features)
    {
        unsigned k = start_index;
        for (unsigned i = start_index; i < end_index; ++i) {
            if (P_[i] < i) {
                P_[i] = P_[P_[i]];
            }
            else {
                P_[i] = k;
		features.Shift<ConfFeatures>(k, i);
                k = k + 1;
            }
        }

	if (end_index > start_index) {
	    end_index = k;
	}
        return k;
    }

    template <typename ConfFeatures>
    unsigned FlattenWithFeatures(Features& features)
    {
        unsigned k = start_index;
        for (unsigned i = start_index; i < end_index; ++i) {
	    unsigned label = P_[i];
            if (label < i) {
		unsigned par = P_[label];
		features.Merge<ConfFeatures>(label, par);
                P_[i] = par;
            }
            else {
                P_[i] = k;
                k = k + 1;
            }
        }

	if (end_index > start_index) {
	    end_index = k;
	}
        return k;
    }

    
    // Flatten a disjoint set of LabelsSolver into a single one
    // This assumes that labels solvers share same memory pool
    static unsigned MultiFlatten(UFPC* ET_chunks, size_t chunk_count) {
	unsigned k = 0;

	assert(chunk_count > 0);
	UFPC* ET = &ET_chunks[0];
	unsigned* __restrict__ P_ = ET->P_;

	k = ET->P_[ET->start_index];
	for (size_t chunk = 0; chunk < chunk_count; chunk++) {
	    UFPC* ET = &ET_chunks[chunk];
	    unsigned start_index = ET->start_index;
	    unsigned end_index = ET->end_index;
	    
	    for (unsigned i = start_index; i < end_index; i++) {
		if (P_[i] < i) {
		    P_[i] = P_[P_[i]];
		} else {
		    P_[i] = k;
		    k++;
		}		
	    }
	    if (end_index > start_index) {
		ET->end_index = k;
	    }
	}
	return k;	
    }

    template <typename ConfFeatures>
    static unsigned MultiFlatten(UFPC* ET_chunks, size_t chunk_count, Features& features) {
	return 0;
    }

    
    unsigned long Size() const {
	return end_index - start_index;
    }
    
    /***************************************************************/

     void MemAlloc(unsigned max_length) {
        mem_P_ = MemVector<unsigned>(max_length);
    }
     void MemDealloc() {}
     void MemSetup() {
        mem_P_[0] = 0;	 // First label is for background pixels
        length_ = 1;
    }
     unsigned MemNewLabel() {
        mem_P_[length_] = length_;
        return length_++;
    }
     unsigned MemGetLabel(unsigned index) {
        return mem_P_[index];
    }

     double MemTotalAccesses() {
        return mem_P_.GetTotalAccesses();
    }

     unsigned MemMerge(unsigned i, unsigned j)
    {
        // FindRoot(i)
        unsigned root(i);
        while (mem_P_[root] < root) {
            root = mem_P_[root];
        }
        if (i != j) {
            // FindRoot(j)
            unsigned root_j(j);
            while (mem_P_[root_j] < root_j) {
                root_j = mem_P_[root_j];
            }
            if (root > root_j) {
                root = root_j;
            }
            // SetRoot(j, root);
            while (mem_P_[j] < j) {
                unsigned t = mem_P_[j];
                mem_P_[j] = root;
                j = t;
            }
            mem_P_[j] = root;
        }
        // SetRoot(i, root);
        while (mem_P_[i] < i) {
            unsigned t = mem_P_[i];
            mem_P_[i] = root;
            i = t;
        }
        mem_P_[i] = root;
        return root;
    }
     unsigned MemFlatten()
    {
        unsigned k = start_index;
        for (unsigned i = start_index; i < end_index; ++i) {
            if (mem_P_[i] < i) {
                mem_P_[i] = mem_P_[mem_P_[i]];
            }
            else {
                mem_P_[i] = k;
                k = k + 1;
            }
        }

	end_index = k + 1;
        return k;
    }

private:
    unsigned *P_;
    unsigned length_;    
    MemVector<unsigned> mem_P_;
    unsigned start_index, end_index;
    bool mem_owner_;
};

// Interleaved Rem algorithm with SPlicing (SP) as in: 
// A New Parallel Algorithm for Two - Pass Connected Component Labeling
// S Gupta, D Palsetia, MMA Patwary
class RemSP {
    // Maximum number of labels (included background) = 2^(sizeof(unsigned) x 8)
public:
     double Alloc(unsigned max_length, PerformanceEvaluator& perf) {
	 Dealloc();
        perf.start();
        P_ = new unsigned[max_length];
        memset(P_, 0, max_length * sizeof(unsigned));
        perf.stop();
        double t = perf.last();
        perf.start();
        memset(P_, 0, max_length * sizeof(unsigned));
        perf.stop();
        return t - perf.last();
    }
    
     void Alloc(unsigned max_length) {
	Dealloc();
        P_ = new unsigned[max_length];
    }
    
     void Dealloc() {
	 if (mem_owner_) {
	     delete[] P_;
	 }
	 P_ = nullptr;
	 mem_owner_ = true;
    }
    
     void Setup() {
        P_[0] = 0;	 // First label is for background pixels
        length_ = 1;
	start_index = 1;
	end_index = 1;
    }

    void SubTable(RemSP& subET, unsigned start_index) {
	subET.P_ = P_;
	subET.P_ = P_;
	subET.start_index = start_index;
	subET.end_index = start_index;
	subET.mem_owner_ = false;	
    }
    
     unsigned NewLabel() {
        P_[end_index] = end_index;
        return end_index++;
    }
    
     unsigned GetLabel(unsigned index) {
        return P_[index];
    }

    unsigned FindRoot(unsigned i) {
	assert(false && "Not implemented yet");
	return 0;
    }
    
    void UpdateTable(unsigned i, unsigned j) {
	assert(false && "Not implemented yet");	
    }

    unsigned long Size() const {
	return end_index - start_index;
    }


    
     unsigned Merge(unsigned i, unsigned j)
    {
        unsigned root_i(i), root_j(j);

        while (P_[root_i] != P_[root_j]) {
            if (P_[root_i] > P_[root_j]) {
                if (root_i == P_[root_i]) {
                    P_[root_i] = P_[root_j];
                    return P_[root_i];
                }
                unsigned z = P_[root_i];
                P_[root_i] = P_[root_j];
                root_i = z;
            }
            else {
                if (root_j == P_[root_j]) {
                    P_[root_j] = P_[root_i];
                    return P_[root_i];
                }
                unsigned z = P_[root_j];
                P_[root_j] = P_[root_i];
                root_j = z;
            }
        }
        return P_[root_i];
    }


    void FlattenNoCompress() {
        for (unsigned i = start_index; i < end_index; ++i) {
            if (P_[i] < i) {
                P_[i] = P_[P_[i]];
            }
	}	
    }

    template <typename ConfFeatures>
    void FlattenNoCompressFeatures(Features& features) {
	for (unsigned i = start_index; i < end_index; ++i) {
	    unsigned par = P_[i];
	    if (par < i) {
		P_[i] = P_[par];
		features.Merge<ConfFeatures>(i, par);
	    }
	}
    }

    
    unsigned Flatten()
    {
        unsigned k = start_index;
        for (unsigned i = start_index; i < end_index; ++i) {
            if (P_[i] < i) {
                P_[i] = P_[P_[i]];
            }
            else {
                P_[i] = k;
                k = k + 1;
            }
        }

	if (end_index > start_index) {
	    end_index = k;
	}
        return k;
    }
    
    template <typename ConfFeatures>
    unsigned Flatten(Features& features)
    {
        unsigned k = start_index;
        for (unsigned i = start_index; i < end_index; ++i) {
            if (P_[i] < i) {
                P_[i] = P_[P_[i]];
            }
            else {
                P_[i] = k;
		features.Shift<ConfFeatures>(k, i);
                k = k + 1;
            }
        }

	if (end_index > start_index) {
	    end_index = k;
	}
        return k;
    }

    template <typename ConfFeatures>
    unsigned FlattenWithFeatures(Features& features) {
	
	unsigned k = start_index;
	for (unsigned i = start_index; i < end_index; ++i) {
	    unsigned label = P_[i];
	    unsigned par = P_[label];

            if (label < i) {
		features.Merge<ConfFeatures>(label, par);
                P_[i] = par;
            }
            else {
		features.Shift<ConfFeatures>(i, k);
                P_[i] = k;		
                k = k + 1;
            }
        }

	if (end_index > start_index) {
	    end_index = k;
	}
        return k;
    }

    
    static unsigned MultiFlatten(RemSP* ET_chunks, size_t chunk_count) {
	return 0;
    }

    template <typename ConfFeatures>
    static unsigned MultiFlatten(RemSP* ET_chunks, size_t chunk_count, Features& features) {
	return 0;
    }
    
    /***************************************************************/

     void MemAlloc(unsigned max_length) {
        mem_P_ = MemVector<unsigned>(max_length);
    }
    
     void MemDealloc() {}
    
     void MemSetup() {
        mem_P_[0] = 0;	 // First label is for background pixels
        length_ = 1;
    }
    
     unsigned MemNewLabel() {
        mem_P_[length_] = length_;
        return length_++;
    }
    
     unsigned MemGetLabel(unsigned index) {
        return mem_P_[index];
    }

     double MemTotalAccesses() {
        return mem_P_.GetTotalAccesses();
    }

     unsigned MemMerge(unsigned i, unsigned j)
    {
        unsigned root_i(i), root_j(j);
        while (mem_P_[root_i] != mem_P_[root_j]) {
            if (mem_P_[root_i] > mem_P_[root_j]) {
                if (root_i == mem_P_[root_i]) {
                    mem_P_[root_i] = mem_P_[root_j];
                    return mem_P_[root_i];
                }
                unsigned z = mem_P_[root_i];
                mem_P_[root_i] = mem_P_[root_j];
                root_i = z;
            }
            else {
                if (root_j == mem_P_[root_j]) {
                    mem_P_[root_j] = mem_P_[root_i];
                    return mem_P_[root_i];
                }
                unsigned z = mem_P_[root_j];
                mem_P_[root_j] = mem_P_[root_i];
                root_j = z;
            }
        }
        return mem_P_[root_i];
    }
     unsigned MemFlatten()
    {
        unsigned k = 1;
        for (unsigned i = 1; i < length_; ++i) {
            if (mem_P_[i] < i) {
                mem_P_[i] = mem_P_[mem_P_[i]];
            }
            else {
                mem_P_[i] = k;
                k = k + 1;
            }
        }
        return k;
    }

private:
    unsigned *P_;
    unsigned length_;
    MemVector<unsigned> mem_P_;
    unsigned start_index, end_index;
    bool mem_owner_;
};

// Three Table Array as in: 
// A Run-Based Two-Scan Labeling Algorithm
// Lifeng He, Yuyan Chao, Kenji Suzuki
class TTA {
    // Maximum number of labels (included background) = 2^(sizeof(unsigned) x 8) - 1:
    // the special value "-1" for next_ table array has been replace with UINT_MAX
public:
    TTA() : mem_owner_(false) {
    }
    
    double Alloc(unsigned max_length, PerformanceEvaluator& perf) {
	Dealloc();
        perf.start();
        rtable_ = new unsigned[max_length];
        next_ = new unsigned[max_length];
        tail_ = new unsigned[max_length];
        memset(rtable_, 0, max_length * sizeof(unsigned));
        memset(next_, 0, max_length * sizeof(unsigned));
        memset(tail_, 0, max_length * sizeof(unsigned));
        perf.stop();
        double t = perf.last();
        perf.start();
        memset(rtable_, 0, max_length * sizeof(unsigned));
        memset(next_, 0, max_length * sizeof(unsigned));
        memset(tail_, 0, max_length * sizeof(unsigned));
        perf.stop();
        return t - perf.last();
    }
    
    void Alloc(unsigned max_length) {
	Dealloc();
        rtable_ = new unsigned[max_length];
        next_ = new unsigned[max_length];
        tail_ = new unsigned[max_length];
    }
    
    void Dealloc() {
	if (mem_owner_) {
	    delete[] rtable_;
	    delete[] next_;
	    delete[] tail_;
	}
	mem_owner_ = true;
	
	rtable_ = nullptr;
	next_ = nullptr;
	tail_ = nullptr;
    }

    void SubTable(TTA& subET, unsigned start_index) {
	subET.mem_owner_ = false;
	subET.rtable_ = rtable_;
	subET.next_ = next_;
	subET.tail_ = tail_;
	subET.length_ = start_index;
	subET.start_index = start_index;
	subET.end_index = start_index;
    }
    
     void Setup() {
	 
        rtable_[0] = 0;
        length_ = 1;
    }
    
     unsigned NewLabel() {
        rtable_[end_index] = end_index;
        next_[end_index] = UINT_MAX;
        tail_[end_index] = end_index;
        return end_index++;
    }
    
    unsigned GetLabel(unsigned index) {
        return rtable_[index];
    }

    unsigned long Size() const {
	return end_index - start_index;
    }

    
    // Basic functions of the TTA solver required only by the Light-Speed Labeling Algorithms:
    // - "UpdateTable" updates equivalences tables without performing "Find" operations
    // - "FindRoot" finds the root of the tree of node i (for the other algorithms it is 
    //		already included in the "Merge" and "Flatten" functions).
     void UpdateTable(unsigned u, unsigned v)
    {
        if (u < v) {
            unsigned i = v;
            while (i != UINT_MAX) {
                rtable_[i] = u;
                i = next_[i];
            }
            next_[tail_[u]] = v;
            tail_[u] = tail_[v];
        }
        else if (u > v) {
            unsigned i = u;
            while (i != UINT_MAX) {
                rtable_[i] = v;
                i = next_[i];
            }
            next_[tail_[v]] = u;
            tail_[v] = tail_[u];
        }
    }
     unsigned FindRoot(unsigned i)
    {
        return rtable_[i];
    }

     unsigned Merge(unsigned u, unsigned v)
    {
        // FindRoot(u);
        u = rtable_[u];
        // FindRoot(v);
        v = rtable_[v];

        if (u < v) {
            unsigned i = v;
            while (i != UINT_MAX) {
                rtable_[i] = u;
                i = next_[i];
            }
            next_[tail_[u]] = v;
            tail_[u] = tail_[v];
            return u;
        }
        else if (u > v) {
            unsigned i = u;
            while (i != UINT_MAX) {
                rtable_[i] = v;
                i = next_[i];
            }
            next_[tail_[v]] = u;
            tail_[v] = tail_[u];
            return v;
        }

        return u;  // equal to v
    }

    void FlattenNoCompress() {
	for (unsigned k = start_index; k < end_index; k++) {
	    unsigned par = rtable_[k];
	    if (par != k) {
                rtable_[k] = rtable_[par];
	    }
	}
    }

    template <typename ConfFeatures>
    void FlattenNoCompressFeatures(Features& features) {
	for (unsigned k = start_index; k < end_index; k++) {
	    unsigned par = rtable_[k];
	    if (par != k) {
                rtable_[k] = rtable_[par];
		features.Merge<ConfFeatures>(k, par);
	    }
	}
    }


    

    
     unsigned Flatten()
    {
        unsigned cur_label = start_index;
        for (unsigned k = start_index; k < end_index; k++) {
            if (rtable_[k] == k) {
                cur_label++;
                rtable_[k] = cur_label;
            }
            else
                rtable_[k] = rtable_[rtable_[k]];
        }

	if (end_index > start_index) {
	    end_index = cur_label;
	}
        return cur_label;
    }

    
    template <typename ConfFeatures>
     unsigned Flatten(Features& features)
    {
        unsigned cur_label = start_index;
        for (unsigned k = start_index; k < end_index; k++) {
            if (rtable_[k] == k) {
                cur_label++;
                rtable_[k] = cur_label;
		features.Shift<ConfFeatures>(cur_label, k);
            }
            else
                rtable_[k] = rtable_[rtable_[k]];
        }

	if (end_index > start_index) {
	    end_index = cur_label;
	}
        return cur_label;
    }

    static unsigned MultiFlatten(TTA* ET_chunks, size_t chunk_count) {
	unsigned cur_label = 0;
	assert(chunk_count > 0);
	unsigned* __restrict__ rtable_ = ET_chunks[0].rtable_;	
	cur_label = ET_chunks[0].start_index;
	
	for (size_t chunk = 0; chunk < chunk_count; chunk++) {
	    TTA* ET = &ET_chunks[chunk];
	    unsigned start_index = ET->start_index;
	    unsigned end_index = ET->end_index;
	    
	    for (unsigned i = start_index; i < end_index; i++) {
		if (rtable_[cur_label] == cur_label) {
		    cur_label++;
		    rtable_[cur_label] = cur_label;
		} else {
		    rtable_[cur_label] = rtable_[rtable_[cur_label]]; 
		}
	    }
	    
	    if (end_index > start_index) {
		ET->end_index = cur_label;
	    }						   
	}
	return cur_label;
    }

    template <typename ConfFeatures>
    static unsigned MultiFlatten(TTA* ET_chunks, size_t chunk_count, Features& features) {
	return 0;
    }

    
    template <typename ConfFeatures>
    unsigned FlattenWithFeatures(Features& features) {
        unsigned cur_label = start_index;
        for (unsigned k = start_index; k < end_index; k++) {
	    int label =- rtable_[k];
            if (label == k) {
                cur_label++;
                rtable_[k] = cur_label;
            }
            else {
		int par = rtable_[label];
		features.Merge<ConfFeatures>(label, par);
                rtable_[k] = par;
	    }
        }

	if (end_index > start_index) {
	    end_index = cur_label;
	}
        return cur_label;
    }

    
    /***************************************************************/

     void MemAlloc(unsigned max_length) {
        mem_rtable_ = MemVector<unsigned>(max_length);
        mem_next_ = MemVector<unsigned>(max_length);
        mem_tail_ = MemVector<unsigned>(max_length);
    }
     void MemDealloc() {}
     void MemSetup() {
        mem_rtable_[0] = 0;
        length_ = 1;
    }
     unsigned MemNewLabel() {
        mem_rtable_[length_] = length_;
        mem_next_[length_] = UINT_MAX;
        mem_tail_[length_] = length_;
        return length_++;
    }
     unsigned MemGetLabel(unsigned index) {
        return mem_rtable_[index];
    }

     double MemTotalAccesses() {
        return mem_rtable_.GetTotalAccesses() +
            mem_next_.GetTotalAccesses() +
            mem_tail_.GetTotalAccesses();
    }

    // Basic functions of the TTA solver required only by the Light-Speed Labeling Algorithms:
    // - "MemUpdateTable" updates equivalences tables without performing "MemFind" operations
    // - "MemFindRoot" finds the root of the tree of node i (for the other algorithms it is 
    //		already included in the "MemMerge" and "MemFlatten" functions).
     void MemUpdateTable(unsigned u, unsigned v)
    {
        if (u < v) {
            unsigned i = v;
            while (i != UINT_MAX) {
                mem_rtable_[i] = u;
                i = mem_next_[i];
            }
            mem_next_[mem_tail_[u]] = v;
            mem_tail_[u] = mem_tail_[v];
        }
        else if (u > v) {
            unsigned i = u;
            while (i != UINT_MAX) {
                mem_rtable_[i] = v;
                i = mem_next_[i];
            }
            mem_next_[mem_tail_[v]] = u;
            mem_tail_[v] = mem_tail_[u];
        }
    }
     unsigned MemFindRoot(unsigned i)
    {
        return mem_rtable_[i];
    }

     unsigned MemMerge(unsigned u, unsigned v)
    {
        // FindRoot(u);
        u = mem_rtable_[u];
        // FindRoot(v);
        v = mem_rtable_[v];

        if (u < v) {
            unsigned i = v;
            while (i != UINT_MAX) {
                mem_rtable_[i] = u;
                i = mem_next_[i];
            }
            mem_next_[mem_tail_[u]] = v;
            mem_tail_[u] = mem_tail_[v];
            return u;
        }
        else if (u > v) {
            unsigned i = u;
            while (i != UINT_MAX) {
                mem_rtable_[i] = v;
                i = mem_next_[i];
            }
            mem_next_[mem_tail_[v]] = u;
            mem_tail_[v] = mem_tail_[u];
            return v;
        }

        return u;  // equal to v
    }
     unsigned MemFlatten()
    {
        // In order to renumber and count the labels: is it really necessary? 
        unsigned cur_label = 1;
        for (unsigned k = 1; k < length_; k++) {
            if (mem_rtable_[k] == k) {
                cur_label++;
                mem_rtable_[k] = cur_label;
            }
            else
                mem_rtable_[k] = mem_rtable_[mem_rtable_[k]];
        }

        return cur_label;
    }

private:
     unsigned *rtable_;
     unsigned *next_;
     unsigned *tail_;
     unsigned length_;
     MemVector<unsigned> mem_rtable_;
     MemVector<unsigned> mem_next_;
     MemVector<unsigned> mem_tail_;
    unsigned start_index, end_index;
    bool mem_owner_;
};

#define REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(algorithm) \
    REGISTER_SOLVER(algorithm, UF) \
	REGISTER_SOLVER(algorithm, UFPC) \
    REGISTER_SOLVER(algorithm, RemSP) \
	REGISTER_SOLVER(algorithm, TTA) \

#endif // !YACCLAB_LABELS_SOLVER_H_

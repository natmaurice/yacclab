#ifndef CCL_ALIGNED_ALLOC_HPP
#define CCL_ALIGNED_ALLOC_HPP

#include <cstddef>
#include <cstdlib>
#include <simdhelpers/utils.hpp>

template <typename T>
T* aligned_new(size_t size, size_t alignment);

template <typename T>
void aligned_delete(T* ptr, size_t alignment);



// Implementations
template <typename T>
T* aligned_new(size_t size, size_t alignment) {
    return (T*)aligned_alloc(alignment, roundup_kpow2(size * sizeof(T), alignment));
}

template <typename T>
void aligned_delete(T* data, size_t alignment) {
    free(data);
}
 

#endif // CCL_ALIGNED_ALLOC_HPP

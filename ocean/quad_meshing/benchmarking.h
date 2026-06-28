#pragma once

#include <stdio.h>
#include <string.h>
#include <time.h>

// Usage:
//   #define BENCHMARKING_ENABLED 1
//   #define BENCHMARKING_PRINT_EVERY 100000ULL
//   #include "benchmarking.h"
//
//   BENCH_START(total, "my.total");
//   BENCH_START(work, "my.work");
//   do_work();
//   BENCH_END(work);
//   BENCH_END(total);
//   BENCH_MAYBE_PRINT(total, "my benchmark");
//
// Tokens must be unique within the current C scope.

#ifndef BENCHMARKING_ENABLED
#define BENCHMARKING_ENABLED 1
#endif

#ifndef BENCHMARKING_MAX_REGIONS
#define BENCHMARKING_MAX_REGIONS 128
#endif

#ifndef BENCHMARKING_PRINT_EVERY
#define BENCHMARKING_PRINT_EVERY 100000ULL
#endif

#if BENCHMARKING_ENABLED

static const char* benchmarking_names[BENCHMARKING_MAX_REGIONS] = {0};
static volatile unsigned long long benchmarking_ns[BENCHMARKING_MAX_REGIONS] = {0};
static volatile unsigned long long benchmarking_calls[BENCHMARKING_MAX_REGIONS] = {0};
static volatile int benchmarking_region_count = 0;
static volatile int benchmarking_registry_lock = 0;
static volatile int benchmarking_printing = 0;

static inline unsigned long long benchmarking_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ULL + (unsigned long long)ts.tv_nsec;
}

static inline unsigned long long benchmarking_load_ull(volatile unsigned long long* ptr) {
    return __sync_fetch_and_add(ptr, 0ULL);
}

static inline int benchmarking_load_int(volatile int* ptr) {
    return __sync_fetch_and_add(ptr, 0);
}

static inline void benchmarking_lock(volatile int* lock) {
    while (__sync_lock_test_and_set(lock, 1)) {}
}

static inline void benchmarking_unlock(volatile int* lock) {
    __sync_lock_release(lock);
}

static inline int benchmarking_register_region(const char* name) {
    benchmarking_lock(&benchmarking_registry_lock);

    int count = benchmarking_load_int(&benchmarking_region_count);
    for (int i = 0; i < count; ++i) {
        if (strcmp(benchmarking_names[i], name) == 0) {
            benchmarking_unlock(&benchmarking_registry_lock);
            return i;
        }
    }

    if (count >= BENCHMARKING_MAX_REGIONS) {
        fprintf(stderr,
            "benchmarking: max regions exceeded (%d). Increase BENCHMARKING_MAX_REGIONS.\n",
            BENCHMARKING_MAX_REGIONS);
        benchmarking_unlock(&benchmarking_registry_lock);
        return -1;
    }

    int id = count;
    benchmarking_names[id] = name;
    __sync_fetch_and_add(&benchmarking_region_count, 1);
    benchmarking_unlock(&benchmarking_registry_lock);
    return id;
}

static inline int benchmarking_region_id_cached(volatile int* cached_id, const char* name) {
    int id = benchmarking_load_int(cached_id);
    if (id >= 0) return id;

    id = benchmarking_register_region(name);
    if (id < 0) return id;
    __sync_bool_compare_and_swap(cached_id, -1, id);
    return benchmarking_load_int(cached_id);
}

static inline void benchmarking_add(int region_id, unsigned long long ns) {
    if (region_id < 0) return;
    __sync_fetch_and_add(&benchmarking_ns[region_id], ns);
    __sync_fetch_and_add(&benchmarking_calls[region_id], 1ULL);
}

static inline void benchmarking_print_region(
        int region_id,
        unsigned long long total_ns,
        unsigned long long total_calls) {
    unsigned long long ns = benchmarking_load_ull(&benchmarking_ns[region_id]);
    unsigned long long calls = benchmarking_load_ull(&benchmarking_calls[region_id]);
    if (calls == 0) return;

    double avg_us_event = (double)ns / (double)calls / 1000.0;
    double avg_us_step = total_calls > 0 ? (double)ns / (double)total_calls / 1000.0 : 0.0;
    double pct = total_ns > 0 ? 100.0 * (double)ns / (double)total_ns : 0.0;
    fprintf(stderr, "  %-32s %9.3f us/event %9.3f us/step %6.2f%% calls=%llu\n",
        benchmarking_names[region_id], avg_us_event, avg_us_step, pct, calls);
}

static inline void benchmarking_maybe_print(int total_region_id, const char* label) {
    if (total_region_id < 0) return;

    unsigned long long total_calls = benchmarking_load_ull(&benchmarking_calls[total_region_id]);
    if (total_calls == 0 || total_calls % BENCHMARKING_PRINT_EVERY != 0) return;
    if (!__sync_bool_compare_and_swap(&benchmarking_printing, 0, 1)) return;

    unsigned long long total_ns = benchmarking_load_ull(&benchmarking_ns[total_region_id]);
    double total_avg_us = total_calls > 0 ? (double)total_ns / (double)total_calls / 1000.0 : 0.0;
    fprintf(stderr,
        "[%s benchmark] calls=%llu avg_total=%.3f us print_every=%llu\n",
        label, total_calls, total_avg_us, (unsigned long long)BENCHMARKING_PRINT_EVERY);

    int count = benchmarking_load_int(&benchmarking_region_count);
    for (int i = 0; i < count; ++i) {
        benchmarking_print_region(i, total_ns, total_calls);
    }
    fprintf(stderr, "\n");
    benchmarking_unlock(&benchmarking_printing);
}

#define BENCH_CONCAT_INNER(a, b) a##b
#define BENCH_CONCAT(a, b) BENCH_CONCAT_INNER(a, b)

#define BENCH_START(token, name) \
    static volatile int BENCH_CONCAT(benchmarking_region_id_, token) = -1; \
    int BENCH_CONCAT(benchmarking_region_, token) = benchmarking_region_id_cached( \
        &BENCH_CONCAT(benchmarking_region_id_, token), (name)); \
    unsigned long long BENCH_CONCAT(benchmarking_start_, token) = benchmarking_now_ns()

#define BENCH_END(token) benchmarking_add( \
    BENCH_CONCAT(benchmarking_region_, token), \
    benchmarking_now_ns() - BENCH_CONCAT(benchmarking_start_, token))

#define BENCH_MAYBE_PRINT(token, label) benchmarking_maybe_print( \
    BENCH_CONCAT(benchmarking_region_, token), (label))

#else

#define BENCH_START(token, name) do { (void)sizeof(name); } while (0)
#define BENCH_END(token) do { } while (0)
#define BENCH_MAYBE_PRINT(token, label) do { (void)sizeof(label); } while (0)

#endif

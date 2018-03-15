#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <vector>

#define VEC_SIZE (int)2e8
#define IT_NUM 1000
void CacheMiss(const std::vector<int> &numbers)
{
    int padded_index = 0;
    int index        = 0;
    int sum          = 0;
    for (int it = 0; it < IT_NUM; it++)
    {
        for (; index < VEC_SIZE; index++)
        {
            padded_index = (padded_index + 64) % VEC_SIZE;
            sum += numbers[padded_index];
        }
    }
}

void CacheHit(const std::vector<int> &numbers)
{
    int padded_index = 0;
    int index        = 0;
    int sum          = 0;
    for (int it = 0; it < IT_NUM; it++)
    {
        for (; index < VEC_SIZE; index++)
        {
            padded_index = (padded_index + 64) % VEC_SIZE;
            sum += numbers[index];
        }
    }
}

int main()
{
    std::vector<int> numbers(VEC_SIZE);

    for (long long i = 0; i < VEC_SIZE; i++)
    {
        numbers[i] = i;
    }

    clock_t start, end;
    double time_diff_hit, time_diff_miss;

    start = clock();
    CacheHit(numbers);
    end = clock();

    time_diff_hit = (end - start) / (double)CLOCKS_PER_SEC;
    printf("Time Hit: %.2lfs \n", time_diff_hit);

    start = clock();
    CacheMiss(numbers);
    end = clock();

    time_diff_miss = (end - start) / (double)CLOCKS_PER_SEC;
    printf("Time Miss: %.2lfs\n", time_diff_miss);

    printf("Using cache is %.2lf%% faster \n",
           (1 - time_diff_hit / time_diff_miss) * 100);
}

#include <cstdio>
#include <cstdlib>
#include <ctime>

#define NUM_IT (int)1e7

void branch_miss()
{
    int sum   = 0, threshold;
    bool flip = true;
    for (int i = 0; i < NUM_IT; i++)
    {
        threshold = rand() % 100;
        flip      = rand() % 100 < threshold;
        if (flip)
            sum++;
        else
            sum--;
    }
}

void branch_hit()
{
    int sum   = 0, threshold;
    bool flip = true;
    for (int i = 0; i < NUM_IT; i++)
    {
        threshold = rand() % 100;
        flip      = rand() % 100 < threshold;
        if (true)
            sum++;
        else
            sum--;
    }
}

int main()
{
    clock_t start, end;
    double time_diff_hit, time_diff_miss;

    start = clock();
    branch_hit();
    end = clock();

    time_diff_hit = (end - start) / (double)CLOCKS_PER_SEC;
    printf("Time Hit: %.3lfs \n", time_diff_hit);

    start = clock();
    branch_miss();
    end = clock();

    time_diff_miss = (end - start) / (double)CLOCKS_PER_SEC;
    printf("Time Miss: %.3lfs\n", time_diff_miss);

    printf("Using branch prediction is %.2lf%% faster \n",
           (1 - time_diff_hit / time_diff_miss) * 100);
}

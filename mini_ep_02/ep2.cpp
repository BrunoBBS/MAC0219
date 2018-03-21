#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

/**
 * Class representing a toad
 */
class Toad
{
public:
    // Constructor
    Toad(int starting_stone, int *stones);

    // Starts thread
    void start_thread();

    // Waits thread
    void wait();

private:
    // Position
    int position;

    // Pointer to stone array
    int *stones;

    // Starts thread to handle this instance
    static void *thread(void *instance);

    // Thread handler
    pthread_t thread_handler;
};

Toad::Toad(int starting_stone, int *stones) :
    position(starting_stone), stones(stones)
{
    stones[position] = this;
}

void Toad::start_thread()
{
    pthread_create(&this->thread_handler, nullptr, &Toad::thread, this);
}

void Toad::wait()
{
    pthread_join(this->thread_handler, nullptr);
}

void *Toad::thread(void *instance)
{
    Toad &toad = *((Toad*) instance);

    for (int i = 0; i < 10; i++)
      printf("Yay %d\n", toad.position);
}

/**
 * Class representing a frog
 */
class Frog
{
};

Toad *toad[500];

int main(int argc, char *argv[])
{
    for (int i = 0; i < 500; i++)
        toad[i] = new Toad(i, nullptr);
    for (int i = 0; i < 500; i++)
        toad[i]->start_thread();
    for (int i = 0; i < 500; i++)
        toad[i]->wait();
}

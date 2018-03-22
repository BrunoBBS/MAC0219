#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
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

Toad::Toad(int starting_stone, int *stones)
    : position(starting_stone), stones(stones)
{
    stones[position] = this;
}

void Toad::start_thread()
{
    pthread_create(&this->thread_handler, nullptr, &Toad::thread, this);
}

void Toad::wait() { pthread_join(this->thread_handler, nullptr); }

void *Toad::thread(void *instance)
{
    Toad &toad = *((Toad *)instance);

    for (int i = 0; i < 10; i++)
        printf("Yay %d\n", toad.position);
}

/**
 * Class representing a frog (start on the left and go to the right)
 */
class Frog
{
  public:
    // Constructor
    Frog(int starting_stone, int *stones);

    // Waits Thread to finalise
    void wait();

  private:
    // Frog's position in the stone array
    int position;

    // Thread handler
    pthread_t thread_handler;

    // Pointer to the stone array
    int *stones;

    /**
     * Function passed to the thread initializer to be the main thread function
     * Receives: a frog instance (since it has to be static), the barrier to
     * sync the start and the semaphore to stomize actions in the stone array
     */
    static void *thread(void *frog_instance, pthread_barrier_t *start,
                        sem_t *stones_semaphore);

    // Function called wehn the frog can jump
    int jump();

    // Funtion that evaluates if Frog can jump
    bool can_jump();
};

Frog::Frog(int starting_stone, int *stones)
    : position(starting_stone), stones(stones)
{
    // TODO: We have to find a way to better represent this
    stones[position] = this;
    pthread_create(&this->thread_handler, nullptr, &Frog::thread, this);
}

void *Frog::thread(void *frog_instance, pthread_barrier_t *start,
                   sem_t *stones_semaphore)
{
    Frog *instance = (Frog *)frog_instance;
    // Waits the program to start
    pthread_barrier_wait(start);

    sem_wait(stones_semaphore);
    if (instance->can_jump()) instance->jump();
    else 
        cant_jump_counter++;
    sem_post(stones_semaphore);
}

int Frog::jump()
{
    stones[position + 1] = this;
    stones[position]     = 0;
}

bool Frog::can_jump()
{
    if (this->stones[position + 1] == 0)
        return true;
    else if (stones[position + 2] == 0)
        return true;
    else
        return false;
}

void Frog::wait() { pthread_join(this->thread_handler, nullptr); }

Toad *toad[500];
int *stones[1000];
sem_t *stones_semaphore;

int main(int argc, char *argv[])
{
    // Initialize the semaphre to atomize the frog jumps
    sem_init(stones_semaphore, 0, 1);

    for (int i = 0; i < 500; i++)
        toad[i] = new Toad(i, nullptr);
    for (int i = 0; i < 500; i++)
        toad[i]->start_thread();
    for (int i = 0; i < 500; i++)
        toad[i]->wait();
}

#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>

/**
 * The stones array receive:
 * 0: free
 * 1: toad
 * 2: frog
 */

// Number of failed jumps the suggest a deadlock
#define DEADLOCK_THRESHOLD 10000
int cant_jump_counter;
sem_t stones_semaphore;
pthread_barrier_t start;

/**
 * Class representing a toad
 */
class Toad
{
  public:
    // Constructor
    Toad(int starting_stone, void **stones);

    // Starts thread
    void start_thread();

    // Waits thread
    void wait();

  private:
    // Position
    int position;

    // Pointer to stone array
    void **stones;

    // Starts thread to handle this instance
    static void *thread(void *instance);

    // Can jump?
    bool can_jump();

    // Jump
    bool jump();

    // Thread handler
    pthread_t thread_handler;
};

Toad::Toad(int starting_stone, void **stones)
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

    pthread_barrier_wait(&start);

    while (true)
    {
        bool jumped = false;
        if (jumped = toad.can_jump())
        {
            sem_wait(&stones_semaphore);
            jumped = toad.jump();
            sem_post(&stones_semaphore);
        }

        if (!jumped)
            cant_jump_counter++;
    }
}

bool Toad::can_jump() {
    bool result = position > 0 && !stones[position - 1];
    result |= position > 1 && !stones[position - 2];
    return result;
}

bool Toad::jump() {
    stones[position] = 0;

    bool jumped = false;
    
    // Check if can jump 2 forward
    if (!jumped && (jumped = (position > 1 && !stones[position - 2])))
        stones[position - 2] = this;
    
    // Check if can jump 1 forward
    if (!jumped && (jumped = (position > 0 && !stones[position - 1])))
        stones[position - 1] = this;

    return jumped;
}

/**
 * Class representing a frog (start on the left and go to the right)
 */
class Frog
{
  public:
    // Constructor
    Frog(int starting_stone, void **stones);

    // Waits Thread to finalise
    void wait();

  private:
    // Frog's position in the stone array
    int position;

    // Thread handler
    pthread_t thread_handler;

    // Pointer to the stone array
    void **stones;

    /**
     * Function passed to the thread initializer to be the main thread function
     * Receives: a frog instance (since it has to be static), the barrier to
     * sync the start and the semaphore to stomize actions in the stone array
     */
    static void *thread(void *frog_instance);

    // Function called wehn the frog can jump
    int jump();

    // Funtion that evaluates if Frog can jump
    bool can_jump();
};

Frog::Frog(int starting_stone, void **stones)
    : position(starting_stone), stones(stones)
{
    // TODO: We have to find a way to better represent this
    stones[position] = this;
    pthread_create(&this->thread_handler, nullptr, &Frog::thread, this);
}

void *Frog::thread(void *frog_instance)
{
    Frog *instance = (Frog *)frog_instance;
    // Waits the program to start
    pthread_barrier_wait(&start);

    while (cant_jump_counter < DEADLOCK_THRESHOLD)
    {
        if (instance->can_jump()) {
            sem_wait(&stones_semaphore);
            if (instance->can_jump())
                instance->jump();
            else
                cant_jump_counter++;
            // TODO: There are two ways to detect a deadlock taht have to be
            // implmented
            sem_post(&stones_semaphore);
        }
        else
            cant_jump_counter++;
    }
}

int Frog::jump()
{
    stones[position + 1] = this;
    stones[position]     = 0;
}

bool Frog::can_jump() { return !stones[position + 1] || !stones[position + 2]; }

void Frog::wait() { pthread_join(this->thread_handler, nullptr); }

/**
 * Where the rest of the program lies
 */
int main(int argc, char *argv[])
{
    cant_jump_counter = 0;
    // Initialize the semaphore to atomize the frog jumps
    sem_init(&stones_semaphore, 0, 1);
}

#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>
#include <string>

// Number of failed jumps the suggest a deadlock
#define DEADLOCK_THRESHOLD 10000
int cant_jump_counter;
sem_t stones_semaphore;
pthread_barrier_t start;

/**
 * Class to represent object - Provides basic identification functionality
 */
class Object
{
  public:
    Object(std::string identifier);

    // Get identifier string
    std::string get_identifier() { return identifier; }

  private:
    std::string identifier;
};

Object::Object(std::string identifier) :
    identifier(identifier)
{
}

/**
 * Threaded objects
 */
class Threaded : public Object
{
  public:
    // Constructor
    Threaded(std::string identifier);

    // Wait for thread to finish
    void wait();

  protected:
    // Virtual function to run on thread
    virtual void run() = 0;

  private:
    // Static function to be called by thread
    static void *thread(void *instance);

    // Thread handler
    pthread_t thread_handler;
};

Threaded::Threaded(std::string identifier) :
    Object(identifier)
{
    pthread_create(&this->thread_handler, nullptr, &Threaded::thread, this);
}

void Threaded::wait() { pthread_join(this->thread_handler, nullptr); }

void *Threaded::thread(void *instance) {
    Threaded &obj = *((Threaded *) instance);
    obj.run();
}

/**
 * Class representing a toad
 */
class Toad : public Threaded
{
  public:
    // Constructor
    Toad(int tid, int starting_stone, Threaded **stones);

  private:
    // Position
    int position;

    // Pointer to stone array
    Threaded **stones;

    // Can jump?
    bool can_jump();

    // Jump
    bool jump();

    // Main logic
    void run();
};

Toad::Toad(int tid, int starting_stone, Threaded **stones)
    : position(starting_stone), stones(stones),
    Threaded("Toad " + std::to_string(tid))
{
    stones[position] = this;
}

void Toad::run()
{
    pthread_barrier_wait(&start);

    while (cant_jump_counter < DEADLOCK_THRESHOLD)
    {
        bool jumped = false;
        if (jumped = can_jump())
        {
            sem_wait(&stones_semaphore);
            jumped = jump();
            sem_post(&stones_semaphore);
        }

        if (!jumped)
            cant_jump_counter++;
        else
            cant_jump_counter = 0;
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
        stones[position -= 2] = this;
    
    // Check if can jump 1 forward
    if (!jumped && (jumped = (position > 0 && !stones[position - 1])))
        stones[position -= 1] = this;

    if (!jumped) stones[position] = this;

    return jumped;
}

/**
 * Class representing a frog (start on the left and go to the right)
 */
class Frog : public Threaded
{
  public:
    // Constructor
    Frog(int fid, int starting_stone, Threaded **stones);

    // Waits Thread to finalise
    void wait();

  private:
    // Frog's position in the stone array
    int position;

    // Pointer to the stone array
    Threaded **stones;

    // Main logic
    void run();

    // Function called wehn the frog can jump
    void jump();

    // Funtion that evaluates if Frog can jump
    bool can_jump();
};

Frog::Frog(int fid, int starting_stone, Threaded **stones)
    : position(starting_stone), stones(stones),
    Threaded("Frog " + std::to_string(fid))
{
    // TODO: We have to find a way to better represent this
    stones[position] = this;
}

void Frog::run()
{
    // Waits the program to start
    pthread_barrier_wait(&start);

    while (cant_jump_counter < DEADLOCK_THRESHOLD)
    {
        if (can_jump()) {
            sem_wait(&stones_semaphore);
            if (can_jump())
                jump();
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

void Frog::jump()
{
    int k                = stones[position + 1] == 0 ? 1 : 2;
    stones[position + k] = this;
    stones[position]     = 0;
    position += k;
}

bool Frog::can_jump() { return !stones[position + 1] || !stones[position + 2]; }

/**
 * Where the rest of the program lies
 */
int main(int argc, char *argv[])
{
    cant_jump_counter = 0;

    // Read values
    int frogs, toads;

    printf("Frogs: ");
    scanf("%d", &frogs);
    
    printf("Toads: ");
    scanf("%d", &toads);

    int stones_cnt = frogs + toads + 1;

    // Initialize the semaphore to atomize the frog jumps
    sem_init(&stones_semaphore, 0, 1);
    pthread_barrier_init(&start, nullptr, frogs + toads + 1);
    
    // Create stones array
    Threaded **stones = new Threaded*[stones_cnt];
    for (int i = 0; i < stones_cnt; i++)
        stones[i] = nullptr;
    
    for (int i = 0; i < frogs; i++)
        new Frog(i, i, stones);
    for (int i = 0; i < toads; i++)
        new Toad(i, stones_cnt - i - 1, stones);

    for (int i = 0; i < stones_cnt; i++)
    {
        if (stones[i]) printf("%s, ", stones[i]->get_identifier().c_str());
        else printf("(nada), ");
    }
    printf("\n");
    pthread_barrier_wait(&start);

    while(true)
    {
        for (int i = 0; i < stones_cnt; i++)
        {
            if (stones[i]) printf("%s, ", stones[i]->get_identifier().c_str());
            else printf("(nada), ");
        }
        printf("\n");
        usleep(1000000);
    }
    
    // Wait for threads to finish
    for (int i = 0; i < stones_cnt; i++)
        if (stones[i]) stones[i]->wait();

    // Cleanup
    for (int i = 0; i < stones_cnt; i++)
        if (stones[i]) delete stones[i];
    delete stones;
}

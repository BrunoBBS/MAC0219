#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <vector>

// Number of failed jumps that suggest a deadlock
int DEADLOCK_ALERT;
sem_t stones_semaphore;
pthread_barrier_t start_b;

/**
 * Class to represent object - Provides basic identification functionality
 *
 */
class Object
{
  public:
    Object(std::string identifier);

    char type;

    // Get identifier string
    std::string get_identifier() { return identifier; }

  private:
    std::string identifier;
};

Object::Object(std::string identifier) : identifier(identifier) {}

/**
 * Threaded objects
 */
class Threaded : public Object
{
  public:
    // Constructor
    Threaded(std::string identifier);

    // Starts thread
    void start();

    // Wait for thread to finish
    void wait();

    virtual bool can_jump() = 0;

  protected:
    // Virtual function to run on thread
    virtual void run() = 0;

  private:
    // Static function to be called by thread
    static void *thread(void *instance);

    // Thread handler
    pthread_t thread_handler;
};

Threaded::Threaded(std::string identifier) : Object(identifier) {}

void Threaded::start()
{
    pthread_create(&thread_handler, nullptr, &Threaded::thread, this);
}

void Threaded::wait() { pthread_join(this->thread_handler, nullptr); }

void *Threaded::thread(void *instance)
{
    Threaded &obj = *((Threaded *)instance);
    obj.run();
}

/**
 * Class representing a toad
 */
class Toad : public Threaded
{
  public:
    // Constructor
    Toad(int tid, int starting_stone, std::vector<Threaded *> &stones);

    // Can jump?
    bool can_jump();

  private:
    // Posiion
    int position;

    // Pointer to stone array
    std::vector<Threaded *> &stones;

    // Jump
    bool jump();

    // Main logic
    void run();
};

Toad::Toad(int tid, int starting_stone, std::vector<Threaded *> &stones)
    : position(starting_stone), stones(stones),
      Threaded("T" + std::to_string(tid))
{
    this->type       = 'T';
    stones[position] = this;
}

void Toad::run()
{
    pthread_barrier_wait(&start_b);

    while (!DEADLOCK_ALERT)
    {
        bool jumped = false;
        if (jumped = can_jump()) {
            sem_wait(&stones_semaphore);
            jumped = jump();
            sem_post(&stones_semaphore);
        }
    }
}

bool Toad::can_jump()
{
    bool result = position > 0 && !stones[position - 1];
    result |= position > 1 && !stones[position - 2];
    return result;
}

bool Toad::jump()
{
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
    Frog(int fid, int starting_stone, std::vector<Threaded *> &stones);

    // Waits Thread to finalise
    void wait();

    // Funtion that evaluates if Frog can jump
    bool can_jump();

  private:
    // Frog's position in the stone array
    int position;

    // Pointer to the stone array
    std::vector<Threaded *> &stones;

    // Main logic
    void run();

    // Function called wehn the frog can jump
    void jump();
};

Frog::Frog(int fid, int starting_stone, std::vector<Threaded *> &stones)
    : position(starting_stone), stones(stones),
      Threaded("F" + std::to_string(fid))
{
    this->type = 'F';
    // TODO: We have to find a way to better represent this
    stones[position] = this;
}

void Frog::run()
{
    // Waits the program to start
    pthread_barrier_wait(&start_b);

    while (!DEADLOCK_ALERT)
    {
        if (can_jump()) {
            sem_wait(&stones_semaphore);
            if (can_jump()) jump();
            sem_post(&stones_semaphore);
        }
    }
}

void Frog::jump()
{
    int k                = stones[position + 1] == 0 ? 1 : 2;
    stones[position + k] = this;
    stones[position]     = 0;
    position += k;
}

bool Frog::can_jump()
{
    return (position + 2 < stones.size() && !stones[position + 2]) ||
           (position + 1 < stones.size() && !stones[position + 1]);
}

/**
 * Overseer function. Used in the overseer thread to check if there
 * TTFF
 */
void *overseer(void *argument)
{
    std::vector<Threaded *> &stones = *((std::vector<Threaded *> *)argument);
    pthread_barrier_wait(&start_b);
    while (!DEADLOCK_ALERT)
    {
        sem_wait(&stones_semaphore);
        DEADLOCK_ALERT = 1;
        for (int i = 0; i < stones.size(); i++)
            if (stones[i] != 0 && stones[i]->can_jump()) DEADLOCK_ALERT = 0;
        sem_post(&stones_semaphore);
    }
}

/**
 * Where the rest of the program lies
 */
int main(int argc, char *argv[])
{
    // Read values
    int frogs, toads;

    printf("Frogs: ");
    scanf("%d", &frogs);

    printf("Toads: ");
    scanf("%d", &toads);

    int stones_cnt = frogs + toads + 1;

    DEADLOCK_ALERT = 0;
    pthread_t thread_overseer;

    // Initialize the semaphore to atomize the frog jumps
    sem_init(&stones_semaphore, 0, 1);
    pthread_barrier_init(&start_b, nullptr, frogs + toads + 1);

    // Create stones array
    std::vector<Threaded *> stones(stones_cnt);

    for (int i = 0; i < frogs; i++)
        (new Frog(i, i, stones))->start();
    for (int i = 0; i < toads; i++)
        (new Toad(i, stones_cnt - i - 1, stones))->start();

    pthread_create(&thread_overseer, nullptr, &overseer, &stones);

    while (!DEADLOCK_ALERT)
    {
        printf("Im printing;\n");
        usleep(100000);
    }

    // Wait for threads to finish
    for (int i = 0; i < stones_cnt; i++)
        if (stones[i]) stones[i]->wait();

    // Cleanup
    for (int i = 0; i < stones_cnt; i++)
        if (stones[i]) delete stones[i];
}

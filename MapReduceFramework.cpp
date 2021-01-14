//
// Created by naamagl on 4/28/19.
//

#include "MapReduceFramework.h"
#include <pthread.h>
#include <cstdio>
#include <atomic>
#include <algorithm>
#include <queue>
#include <iostream>
#include <semaphore.h>
#include "Barrier.h"

using namespace std;
typedef struct JobContext
{
    std::atomic<int>* atomic_count;
    JobState *state;
    pthread_t *threads;
    int num_of_threads;
    int inter_vec_size = 0;
    int reduce_done = 0;
    int count_stage_done = 0;
    const InputVec *input_vector;
    const MapReduceClient *client;
    OutputVec *output_vector;
    std::vector<IntermediateVec*>* all_threads_vectors;
    Barrier* barrier;
    bool shuffling = false;
    bool finished_shuffling = false;
    bool in_map = false;
    bool has_joined = false;
    pthread_mutex_t mapping_mutex;
    pthread_mutex_t map_percentage_mutex;
    pthread_mutex_t map_update_mutex;
    pthread_mutex_t shuffle_mutex;
    pthread_mutex_t reduce_init_mutex;
    pthread_mutex_t reduce_inter_mutex;
    pthread_mutex_t reduce_update_mutex;
    pthread_mutex_t emit3_mutex;
    pthread_mutex_t get_state_mutex;
    pthread_mutex_t join_mutex;
    queue<IntermediateVec>* shuffled;
    sem_t shuffle_reduce_sem;


} JobContext;

/**
 * compares between 2 Intermediate Pairs
 */
bool compare_keys(IntermediatePair pair1, IntermediatePair pair2)
{
    const K2* temp = pair2.first;
    return pair1.first->operator<(*temp);
}

/**
 * finds that max K2 key of IntermediateVecs
 * @param v vector of IntermediateVecs
 * @return the max key
 */
K2* find_max_key(std::vector<IntermediateVec*>* v){
    auto iter = v->begin();
    while((*iter)->empty()){
        iter++;
    }
    K2 *max = (*iter)->back().first;
    for (auto i = v->begin(); i != v->end(); i++){
        if (!(*i)->empty()) {
            const K2 *temp = (*i)->back().first;
            if (max->operator<(*temp)) {
                max = (*i)->back().first;
            }
        }
    }
    return max;
}


/**
 * adds the key,value pair to the Intermediate vector
 * @param key
 * @param value
 * @param context
 */
void emit2 (K2* key, V2* value, void* context){
    IntermediateVec* vec = (IntermediateVec*) context;
    IntermediatePair pair(key,value);
    vec->push_back(pair);
}

/**
 * adds the key,value pair to the Output vector
 * @param key
 * @param value
 * @param context
 */
void emit3 (K3* key, V3* value, void* context){
//    OutputVec* vec = (OutputVec*) context;
    JobContext* jc = (JobContext*) context;
    OutputPair pair(key,value);
    if (pthread_mutex_lock(&jc->emit3_mutex) != 0){
        fprintf(stderr, "error on pthread_mutex_lock");
        exit(1);
    }
    jc->output_vector->push_back(pair);
    if (pthread_mutex_unlock(&jc->emit3_mutex) != 0){
        fprintf(stderr, "error on pthread_mutex_unlock");
        exit(1);
    }
}


/**
 * runs the map stage
 * @param jc
 * @param inter_single_thread_vec
 */
void mapPhase(JobContext* jc, IntermediateVec* inter_single_thread_vec) {

    if (pthread_mutex_lock(&jc->mapping_mutex) != 0){
        fprintf(stderr, "error on pthread_mutex_lock");
        exit(1);
    }
    jc->all_threads_vectors->push_back(inter_single_thread_vec);
    int old_val = (*(jc->atomic_count)).load();
    (*(jc->atomic_count))++;

    if (old_val < (int)jc->input_vector->size())
    {
        //update state to map
        if (!jc->in_map) {
            jc->state->stage = MAP_STAGE;
            jc->state->percentage = 0;
            jc->in_map = true;
        }

    }
    if (pthread_mutex_unlock(&jc->mapping_mutex) != 0){
        fprintf(stderr, "error on pthread_mutex_unlock");
        exit(1);
    }

    //sends to map
    while (old_val < (int)jc->input_vector->size()){
        jc->client->map((*jc->input_vector)[old_val].first, (*jc->input_vector)[old_val].second, inter_single_thread_vec);

        //update percentage
        if (pthread_mutex_lock(&jc->map_percentage_mutex) != 0){
            fprintf(stderr, "error on pthread_mutex_lock");
            exit(1);
        }
        (jc->count_stage_done)++;
        jc->state->percentage = (float)jc->count_stage_done*100/(jc->input_vector->size());
        if (pthread_mutex_unlock(&jc->map_percentage_mutex) != 0){
            fprintf(stderr, "error on pthread_mutex_unlock");
            exit(1);
        }
        if (pthread_mutex_lock(&jc->map_update_mutex) != 0){
            fprintf(stderr, "error on pthread_mutex_lock");
            exit(1);
        }
        old_val = (*(jc->atomic_count)).load();
        (*(jc->atomic_count))++;
        if (pthread_mutex_unlock(&jc->map_update_mutex) != 0){
            fprintf(stderr, "error on pthread_mutex_unlock");
            exit(1);
        }

    }
}


/**
 * runs the shuffle stage by the first thread who finished mapping
 * @param jc
 */
void shufflePhase(JobContext* jc) {
    jc->shuffling = true;
    jc->state->stage = REDUCE_STAGE;
    jc->state->percentage = 0;

    for (int i = 0; i < (int)jc->all_threads_vectors->size(); ++i) {
        jc->inter_vec_size+= (*jc->all_threads_vectors)[i]->size();
    }
    int cur_size = jc->inter_vec_size;
    if (pthread_mutex_unlock(&jc->shuffle_mutex) != 0){
        fprintf(stderr, "error on pthread_mutex_unlock");
        exit(1);
    }

    while (cur_size > 0) {
        const K2 *temp_key = find_max_key(jc->all_threads_vectors);
        IntermediateVec temp_key_vec;

        for (int i = 0; i < (int)jc->all_threads_vectors->size(); ++i)
        {
            if (!(*jc->all_threads_vectors)[i]->empty()) {
                const K2 *curr_pos = (*jc->all_threads_vectors)[i]->back().first;
                while (! (*jc->all_threads_vectors)[i]->empty() &&
                    ! (curr_pos->operator<(*temp_key))
                    && ! (temp_key->operator<(*curr_pos))) {
                    temp_key_vec.push_back((*jc->all_threads_vectors)[i]->back());
                    (*jc->all_threads_vectors)[i]->pop_back();
                    cur_size --;
                    curr_pos = (*jc->all_threads_vectors)[i]->back().first;

                }
            }
        }
        if (pthread_mutex_lock(&jc->reduce_inter_mutex) != 0){
            fprintf(stderr, "error on pthread_mutex_lock");
            exit(1);
        }
        jc->shuffled->push(temp_key_vec);
        if (pthread_mutex_unlock(&jc->reduce_inter_mutex) != 0){
            fprintf(stderr, "error on pthread_mutex_unlock");
            exit(1);
        }
        sem_post(&jc->shuffle_reduce_sem);

    }
    jc->finished_shuffling = true;
}


/**
 * runs the reduce stage
 * @param jc
 */
void reducePhase(JobContext* jc){
    if (pthread_mutex_lock(&jc->reduce_init_mutex) != 0){
        fprintf(stderr, "error on pthread_mutex_lock");
        exit(1);
    }

    if (pthread_mutex_unlock(&jc->reduce_init_mutex) != 0){
        fprintf(stderr, "error on pthread_mutex_unlock");
        exit(1);
    }
    while (!jc->finished_shuffling || !jc->shuffled->empty())
    {
        sem_wait(&jc->shuffle_reduce_sem);
        if (pthread_mutex_lock(&jc->reduce_inter_mutex) != 0){
            fprintf(stderr, "error on pthread_mutex_lock");
            exit(1);
        }
        if (!jc->finished_shuffling || !jc->shuffled->empty())
        {
            IntermediateVec temp_shuffled = jc->shuffled->front();
            jc->shuffled->pop();
            if (pthread_mutex_unlock(&jc->reduce_inter_mutex) != 0) {
                fprintf(stderr, "error on pthread_mutex_unlock");
                exit(1);
            }
            jc->client->reduce(&temp_shuffled, jc);

            if (pthread_mutex_lock(&jc->reduce_update_mutex) != 0) {
                fprintf(stderr, "error on pthread_mutex_lock");
                exit(1);
            }
            jc->reduce_done+= temp_shuffled.size();
            jc->state->percentage = (float)jc->reduce_done*100/jc->inter_vec_size;
            if (pthread_mutex_unlock(&jc->reduce_update_mutex) != 0) {
                fprintf(stderr, "error on pthread_mutex_unlock");
                exit(1);
            }
        }
        else{
            if (pthread_mutex_unlock(&jc->reduce_inter_mutex) != 0) {
                fprintf(stderr, "error on pthread_mutex_unlock");
                exit(1);
            }
        }
    }

    for(int i=0 ; i< jc->num_of_threads; ++i)
    {
        sem_post(&jc->shuffle_reduce_sem);
    }

}


/**
 * the function given to each new thread, executes map and reduce for each thread
 * @param arg
 * @return
 */
void* threadAction(void *arg){
    JobContext* jc = (JobContext*) arg;
    IntermediateVec* inter_single_thread_vec = new IntermediateVec;
    mapPhase(jc, inter_single_thread_vec);

    std::sort(inter_single_thread_vec->begin(), inter_single_thread_vec->end(), compare_keys);
    jc->barrier->barrier();

    //shuffle
    if (pthread_mutex_lock(&jc->shuffle_mutex) != 0) {
        fprintf(stderr, "error on pthread_mutex_lock");
        exit(1);
    }
    if (!jc->shuffling) {
        shufflePhase(jc);
    }
    else{
        if (pthread_mutex_unlock(&jc->shuffle_mutex) != 0) {
            fprintf(stderr, "error on pthread_mutex_unlock");
            exit(1);
        }
    }

    //reduce
    reducePhase(jc);
    return 0;
}


/**
 * starts the program per job
 * @param client
 * @param inputVec
 * @param outputVec
 * @param multiThreadLevel
 * @return
 */
JobHandle startMapReduceJob(const MapReduceClient& client,const InputVec& inputVec, OutputVec& outputVec,
        int multiThreadLevel){

    pthread_t *threads = new pthread_t[multiThreadLevel];
    JobContext *job_context = new JobContext;
    Barrier *barrier = new Barrier(multiThreadLevel);

    job_context->threads = threads;
    job_context->num_of_threads = multiThreadLevel;
    job_context->state = new JobState{UNDEFINED_STAGE, 0};
    job_context->barrier = barrier;
    job_context->all_threads_vectors = new std::vector<IntermediateVec*>;
    std::atomic<int> *atomic_counter = new std::atomic<int>(0);
    job_context->atomic_count = atomic_counter;
    job_context->input_vector = &inputVec;
    job_context->output_vector = &outputVec;
    job_context->client = &client;
    job_context->map_update_mutex =PTHREAD_MUTEX_INITIALIZER;
    job_context->mapping_mutex = PTHREAD_MUTEX_INITIALIZER;
    job_context->map_percentage_mutex = PTHREAD_MUTEX_INITIALIZER;
    job_context->shuffle_mutex = PTHREAD_MUTEX_INITIALIZER;
    job_context->reduce_init_mutex = PTHREAD_MUTEX_INITIALIZER;
    job_context->reduce_inter_mutex = PTHREAD_MUTEX_INITIALIZER;
    job_context->reduce_update_mutex = PTHREAD_MUTEX_INITIALIZER;
    job_context->emit3_mutex = PTHREAD_MUTEX_INITIALIZER;
    job_context->get_state_mutex = PTHREAD_MUTEX_INITIALIZER;
    job_context->join_mutex = PTHREAD_MUTEX_INITIALIZER;

    job_context->shuffled = new queue<IntermediateVec>;

    for (int i = 0; i < multiThreadLevel; ++i){
        pthread_create(threads + i, nullptr, threadAction, job_context);
    }

    if (sem_init(&(job_context->shuffle_reduce_sem),0, 0) != 0)
    {
        fprintf(stderr, "error in semaphore");
        exit(1);
    }

    JobHandle job_handler = (JobHandle)job_context;
    return job_handler;
}


/**
 * waits for other threads to finish their work
 * @param job
 */
void waitForJob(JobHandle job) {
    JobContext *jc = (JobContext *) job;
    pthread_mutex_lock(&jc->join_mutex);
    if (!jc->has_joined) {
        for (int i = 0; i < jc->num_of_threads; ++ i) {
            pthread_join(jc->threads[i], NULL);
        }
        jc->has_joined = true;
    }
    pthread_mutex_unlock(&jc->join_mutex);
}


/**
 * sets the given state to the current state of the job
 * @param job
 * @param state
 */
void getJobState(JobHandle job, JobState* state){
    JobContext* jc = (JobContext*) job;
    pthread_mutex_lock(&jc->get_state_mutex);
    state->stage = jc->state->stage;
    state->percentage = jc->state->percentage;
    pthread_mutex_unlock(&jc->get_state_mutex);
}


/**
 * deletes all mutexes
 * @param job_context
 */
void delete_mutexes(JobContext* job_context){

    if (pthread_mutex_destroy(&job_context->map_update_mutex) != 0) {
        fprintf(stderr, "error on pthread_mutex_destroy");
        exit(1);
    }

    if (pthread_mutex_destroy(&job_context->mapping_mutex) != 0) {
        fprintf(stderr, "error on pthread_mutex_destroy");
        exit(1);
    }

    if (pthread_mutex_destroy(&job_context->map_percentage_mutex) != 0) {
        fprintf(stderr, "error on pthread_mutex_destroy");
        exit(1);
    }

    if (pthread_mutex_destroy(&job_context->shuffle_mutex) != 0) {
        fprintf(stderr, "error on pthread_mutex_destroy");
        exit(1);
    }

    if (pthread_mutex_destroy(&job_context->reduce_init_mutex) != 0) {
        fprintf(stderr, "error on pthread_mutex_destroy");
        exit(1);
    }

    if (pthread_mutex_destroy(&job_context->reduce_inter_mutex) != 0) {
        fprintf(stderr, "error on pthread_mutex_destroy");
        exit(1);
    }

    if (pthread_mutex_destroy(&job_context->reduce_update_mutex) != 0) {
        fprintf(stderr, "error on pthread_mutex_destroy");
        exit(1);
    }

    if (pthread_mutex_destroy(&job_context->emit3_mutex) != 0) {
        fprintf(stderr, "error on pthread_mutex_destroy");
        exit(1);
    }

    if (pthread_mutex_destroy(&job_context->get_state_mutex) != 0) {
        fprintf(stderr, "error on pthread_mutex_destroy");
        exit(1);
    }

    if (pthread_mutex_destroy(&job_context->join_mutex) != 0) {
        fprintf(stderr, "error on pthread_mutex_destroy");
        exit(1);
    }

}


/**
 * releases all used resources
 * @param job
 */
void closeJobHandle(JobHandle job){
    waitForJob(job);
    JobContext* jc = (JobContext*) job;
    for (auto i = jc->all_threads_vectors->begin(); i != jc->all_threads_vectors->end(); ++i){
        delete *i;
    }
    delete jc->all_threads_vectors;
    delete jc->threads;
    delete jc->barrier;
    delete jc->atomic_count;
    delete jc->shuffled;
    delete jc->state;
    delete_mutexes(jc);
    delete jc;
}

/* ------------
 * This code is provided solely for the personal and private use of 
 * students taking the CSC367 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * 
 * Authors: Bogdan Simion, Maryam Dehnavi, Felipe de Azevedo Piovezan
 * 
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2020 Bogdan Simion and Maryam Dehnavi
 * -------------
*/

#include "filters.h"
#include <pthread.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

typedef struct common_work_t
    {
        const filter *f;
        const int32_t *original_image;
        int32_t *output_image;
        int32_t width;
        int32_t height;
        int32_t max_threads;
        pthread_barrier_t barrier;
} common_work;

typedef struct work_t
    {
        common_work *common;
        int32_t id;
} work;

typedef struct chunk_t
{
    int32_t row;
    int32_t col;
    int32_t w_chunk;
} chunk;

int min_pix = INT_MAX;
int max_pix = INT_MIN;

pthread_mutex_t min_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t max_mutex = PTHREAD_MUTEX_INITIALIZER;

pthread_mutex_t q_mutex = PTHREAD_MUTEX_INITIALIZER;

chunk *q;
int32_t q_size = 0;
int32_t q_index = 0;
int32_t norm_index = 0;

/************** FILTER CONSTANTS*****************/
/* laplacian */
int8_t lp3_m[] =
    {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    };
filter lp3_f = {3, lp3_m};

int8_t lp5_m[] =
    {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
    };
filter lp5_f = {5, lp5_m};

/* Laplacian of gaussian */
int8_t log_m[] =
    {
        0, 1, 1, 2, 2, 2, 1, 1, 0,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        2, 5, 0, -24, -40, -24, 0, 5, 2,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        0, 1, 1, 2, 2, 2, 1, 1, 0,
    };
filter log_f = {9, log_m};

/* Identity */
int8_t identity_m[] = {1};
filter identity_f = {1, identity_m};

filter *builtin_filters[NUM_FILTERS] = {&lp3_f, &lp5_f, &log_f, &identity_f};

/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest, 
        int32_t largest)
{
    if (smallest == largest)
    {
        return;
    }
    
    target[pixel_idx] = ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}

int my_max(int a, int b) {
    return (a > b) ? a: b;
}

int my_min(int a, int b) {
    return (a < b) ? a : b;
}
/*************** COMMON WORK ***********************/
/* Process a single pixel and returns the value of processed pixel
 * TODO: you don't have to implement/use this function, but this is a hint
 * on how to reuse your code.
 * */
int32_t apply2d(const filter *f, const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int row, int column)
{
    int curr_col;
    int sum = 0;
    int offset = f->dimension / 2;

    int curr_row = row - offset;

    // loop through laplacian filter
    for (int m = 0; m < f->dimension; m++)
    {
        // row has to be valid
        curr_col = column - offset;
        if (curr_row < 0 || curr_row >=height) {
            curr_row += 1;
            continue;
        }
        for (int n = 0; n < f->dimension; n++)
        {
            if (curr_col >= 0 && curr_col < width) {
                int original_lookup = (curr_row * width) + curr_col;
                int filter_lookup = (m * f->dimension) + n;
                sum += f->matrix[filter_lookup] * original[original_lookup];
            }
            curr_col += 1;
        }
        curr_row += 1;
        
    }
    int target_lookup = (row*width) + column;
    target[target_lookup] = sum;
    
    return sum;
}

/*********SEQUENTIAL IMPLEMENTATIONS ***************/
/* TODO: your sequential implementation goes here.
 * IMPORTANT: you must test this thoroughly with lots of corner cases and 
 * check against your own manual calculations on paper, to make sure that your code
 * produces the correct image.
 * Correctness is CRUCIAL here, especially if you re-use this code for filtering 
 * pieces of the image in your parallel implementations! 
 */
void apply_filter2d(const filter *f, 
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height)
{
    int val;
    int entries = height * width;
    // loop through all entries of the matrix and apply2d
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            /* code */
            val = apply2d(f,original,target,width,height,i,j);
            min_pix = my_min(min_pix, val);
            max_pix = my_max(max_pix, val);
        }
        
    }

    for (int i = 0; i < entries; i++) {
        normalize_pixel(target, i, min_pix, max_pix);
    }
}

/****************** ROW/COLUMN SHARDING ************/
/* TODO: you don't have to implement this. It is just a suggestion for the
 * organization of the code.
 */

/* Recall that, once the filter is applied, all threads need to wait for
 * each other to finish before computing the smallest/largets elements
 * in the resulting matrix. To accomplish that, we declare a barrier variable:
 *      pthread_barrier_t barrier;
 * And then initialize it specifying the number of threads that need to call
 * wait() on it:
 *      pthread_barrier_init(&barrier, NULL, num_threads);
 * Once a thread has finished applying the filter, it waits for the other
 * threads by calling:
 *      pthread_barrier_wait(&barrier);
 * This function only returns after *num_threads* threads have called it.
 */
void* sharding_work(void *work)
{
    /* Your algorithm is essentially:
     *  1- Apply the filter on the image
     *  2- Wait for all threads to do the same
     *  3- Calculate global smallest/largest elements on the resulting image
     *  4- Scale back the pixels of the image. For the non work queue
     *      implementations, each thread should scale the same pixels
     *      that it worked on step 1.
     */
    return NULL;
}

void* row_sharded_work(void *w) {
    work *work_struct = (work*) w;
    int work_chunk = work_struct->common->height / work_struct->common->max_threads;
    int curr_row = work_struct->id * work_chunk;
    int end_row = curr_row + work_chunk;

    int local_min = INT_MAX;
    int local_max = INT_MIN;
    end_row += (work_struct->id == work_struct->common->max_threads-1) ? (work_struct->common->height % work_struct->common->max_threads) : 0;
    
    for (int i = curr_row; i < end_row; i++)
    {
        for (int j = 0; j < work_struct->common->width; j++)
        {
            /* code */
            apply2d(work_struct->common->f,
                    work_struct->common->original_image,
                    work_struct->common->output_image,
                    work_struct->common->width,
                    work_struct->common->height,
                    i,
                    j);
            int output_lookup = (i * work_struct->common->width) + j;
            local_min = my_min(local_min, work_struct->common->output_image[output_lookup]);
            local_max = my_max(local_max, work_struct->common->output_image[output_lookup]);
        }
        
    }
    pthread_barrier_wait(&work_struct->common->barrier);

    pthread_mutex_lock(&min_mutex);
    min_pix = my_min(local_min, min_pix);
    pthread_mutex_unlock(&min_mutex);

    pthread_mutex_lock(&max_mutex);
    max_pix = my_max(local_max, max_pix);
    pthread_mutex_unlock(&max_mutex);

    pthread_barrier_wait(&work_struct->common->barrier);
    
    for (int i = curr_row; i < end_row; i++)
    {
        for (int j = 0; j < work_struct->common->width; j++)
        {
            /* code */
            int output_lookup = (i * work_struct->common->width) + j; 
            normalize_pixel(work_struct->common->output_image, output_lookup, min_pix, max_pix);
        }
    }
    return NULL;
}

void* sharding_row_major(void *workload){
    /* Your algorithm is essentially:
     *  1- Apply the filter on the image
     *  2- Wait for all threads to do the same
     *  3- Calculate global smallest/largest elements on the resulting image
     *  4- Scale back the pixels of the image. For the non work queue
     *      implementations, each thread should scale the same pixels
     *      that it worked on step 1.
     */
    
    // sharded column column major
    int local_min = INT_MAX;
    int local_max = INT_MIN;
    int val;
    work *arg = (work *) workload;
    common_work *common = arg->common;
 
 
    // step 1, loop through all entries of the chunk and apply filter
 
    int chunk = common->width / common->max_threads;
    int col = chunk * arg->id;
 
    int col_end = col + chunk;
    if (arg->id == common->max_threads - 1) col_end = col_end + (common->width % common->max_threads);
 
    for (int i = 0; i < common->height; i++){
        for(int j = col; j < col_end; j++){
            val = apply2d(common->f, common->original_image, common->output_image, common->width, common->height, i, j);
            local_max = my_max(val, local_max);
            local_min = my_min(val, local_min);
        }
    }
 
    // step 2, waiting on the pthreads to finish    
    pthread_barrier_wait(&common->barrier);
 
    // step 3, calculate global min and max
 
    pthread_mutex_lock(&min_mutex);
    min_pix = my_min(local_min, min_pix);
    pthread_mutex_unlock(&min_mutex);
 
    pthread_mutex_lock(&max_mutex);
    max_pix = my_max(local_max, max_pix);
    pthread_mutex_unlock(&max_mutex);
    
    pthread_barrier_wait(&common->barrier);
 
    // step 4, normalize given chunk pixels 
 
    for (int i = col; i < col_end; i++){
        for(int j = 0; j < common->height; j++){
            normalize_pixel(common->output_image, j*common->width + i, min_pix, max_pix);
        }
    }

    return NULL;
}

void* sharding_column_major(void *workload)
{
    /* Your algorithm is essentially:
     *  1- Apply the filter on the image
     *  2- Wait for all threads to do the same
     *  3- Calculate global smallest/largest elements on the resulting image
     *  4- Scale back the pixels of the image. For the non work queue
     *      implementations, each thread should scale the same pixels
     *      that it worked on step 1.
     */
    
    // sharded column column major
    int local_min = INT_MAX;
    int local_max = INT_MIN;
    int val;
    work *arg = (work *) workload;
    common_work *common = arg->common;
 
 
    // step 1, loop through all entries of the chunk and apply filter
 
    int chunk = common->width / common->max_threads;
    int col = chunk * arg->id;
 
    int col_end = col + chunk;
    if (arg->id == common->max_threads - 1) col_end = col_end + (common->width % common->max_threads);
 
    for (int i = col; i < col_end; i++){
        for(int j = 0; j < common->height; j++){
            val = apply2d(common->f, common->original_image, common->output_image, common->width, common->height, j, i);
            local_max = my_max(val, local_max);
            local_min = my_min(val, local_min);
        }
    }
 
    // step 2, waiting on the pthreads to finish    
    pthread_barrier_wait(&common->barrier);
 
    // step 3, calculate global min and max
 
    pthread_mutex_lock(&min_mutex);
    min_pix = my_min(local_min, min_pix);
    pthread_mutex_unlock(&min_mutex);
 
    pthread_mutex_lock(&max_mutex);
    max_pix = my_max(local_max, max_pix);
    pthread_mutex_unlock(&max_mutex);
    
    pthread_barrier_wait(&common->barrier);
 
    // step 4, normalize given chunk pixels 
 
    for (int i = col; i < col_end; i++){
        for(int j = 0; j < common->height; j++){
            normalize_pixel(common->output_image, j*common->width + i, min_pix, max_pix);
        }
    }
 
    return NULL;
}

/***************** WORK QUEUE *******************/
/* TODO: you don't have to implement this. It is just a suggestion for the
 * organization of the code.
 */
void* queue_work(void *work)
{
    common_work *common = (common_work*) work;
    int local_min = INT_MAX;
    int local_max = INT_MIN;

    pthread_mutex_lock(&q_mutex);

    while (q_index < q_size) {
        chunk c = q[q_index];
        q_index += 1;
        pthread_mutex_unlock(&q_mutex);
        // do the work
        for (int curr_row = c.row; curr_row < c.row + c.w_chunk; curr_row++)
        {
            /* code */
            for (int curr_col = c.col; curr_col < c.col + c.w_chunk; curr_col++)
            {
                int val = apply2d(common->f,
                    common->original_image,
                    common->output_image,
                    common->width,
                    common->height,
                    curr_row,
                    curr_col);
                local_min = my_min(local_min, val);
                local_max = my_max(local_max, val);
            }
        }

        pthread_mutex_lock(&max_mutex);
        max_pix = my_max(max_pix, local_max);
        min_pix = my_min(min_pix, local_min);
        pthread_mutex_unlock(&max_mutex);

        pthread_mutex_lock(&q_mutex);
    }
    pthread_mutex_unlock(&q_mutex);

    pthread_barrier_wait(&common->barrier);

    pthread_mutex_lock(&q_mutex);
    while (norm_index < q_size) {
        chunk c = q[norm_index];
        norm_index += 1;
        pthread_mutex_unlock(&q_mutex);
        // do the work
        for (int curr_row = c.row; curr_row < c.row + c.w_chunk; curr_row++)
        {
            /* code */
            for (int curr_col = c.col; curr_col < c.col + c.w_chunk; curr_col++)
            {
                int output_lookup = (curr_row * common->width) + curr_col; 
                normalize_pixel(common->output_image, output_lookup, min_pix, max_pix);
            }
        }
        pthread_mutex_lock(&q_mutex);
    }
    pthread_mutex_unlock(&q_mutex);
    return NULL;
}

/***************** MULTITHREADED ENTRY POINT ******/
/* TODO: this is where you should implement the multithreaded version
 * of the code. Use this function to identify which method is being used
 * and then call some other function that implements it.
 */
void apply_filter2d_threaded(const filter *f,
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int32_t num_threads, parallel_method method, int32_t work_chunk)
{
    /* You probably want to define a struct to be passed as work for the
     * threads.
     * Some values are used by all threads, while others (like thread id)
     * are exclusive to a given thread. For instance:
     *   typedef struct common_work_t
     *   {
     *       const filter *f;
     *       const int32_t *original_image;
     *       int32_t *output_image;
     *       int32_t width;
     *       int32_t height;
     *       int32_t max_threads;
     *       pthread_barrier_t barrier;
     *   } common_work;
     *   typedef struct work_t
     *   {
     *       common_work *common;
     *       int32_t id;
     *   } work;
     *
     * An uglier (but simpler) solution is to define the shared variables
     * as global variables.
     */

    common_work* c_work = malloc(sizeof(common_work));
    c_work->f = f;
    c_work->original_image = original;
    c_work->output_image = target;
    c_work->width = width;
    c_work->height = height;
    c_work->max_threads = num_threads;
    pthread_barrier_init(&c_work->barrier, NULL, num_threads);

    pthread_t tids[num_threads];
    int indexes[num_threads];
    for (int i = 0; i < num_threads; i++)
    {
        /* code */
        indexes[i] = i;
    }
    
    switch(method) {
        case SHARDED_ROWS:
        {
            for (int i = 0; i < num_threads; i++)
            {
                work *w = malloc(sizeof(work));
                w->common = c_work;
                w->id = indexes[i];

                pthread_create(&tids[i], NULL, row_sharded_work, (void *) w);
            }

            for (int i = 0; i < num_threads; i++)
            {
                pthread_join(tids[i], NULL);
            }            
            break;
        }
        case SHARDED_COLUMNS_COLUMN_MAJOR:
        {
            for (int i = 0; i < num_threads; i++)
            {
                work *w = malloc(sizeof(work));
                w->common = c_work;
                w->id = indexes[i];

                pthread_create(&tids[i], NULL, sharding_column_major, (void *) w);
            }

            for (int i = 0; i < num_threads; i++)
            {
                pthread_join(tids[i], NULL);
            }            
            break;
        }
        case SHARDED_COLUMNS_ROW_MAJOR:
        {
            for (int i = 0; i < num_threads; i++)
            {
                work *w = malloc(sizeof(work));
                w->common = c_work;
                w->id = indexes[i];

                pthread_create(&tids[i], NULL, sharding_row_major, (void *) w);
            }

            for (int i = 0; i < num_threads; i++)
            {
                pthread_join(tids[i], NULL);
            }            
            break;
        }
        case WORK_QUEUE:
        {
            int32_t col_divide = 1 + ((width - 1)  / work_chunk);
            int32_t row_divide = 1 + ((height -1) / work_chunk);
            q_size = col_divide * row_divide;
            int idx = 0;
            q = malloc(sizeof(chunk) * q_size);
            for (int j = 0; j < row_divide; j++)
            {
                for (int i = 0; i < col_divide; i++)
                {
                    chunk c = {.row=work_chunk*j, .col=work_chunk*i, .w_chunk=work_chunk};
                    q[idx] = c;
                    idx += 1;
                }
            }
            
            for (int i = 0; i < num_threads; i++)
            {

                pthread_create(&tids[i], NULL, queue_work, (void *) c_work);
            }

            for (int i = 0; i < num_threads; i++)
            {
                pthread_join(tids[i], NULL);
            }
            break;
        }
    }
}

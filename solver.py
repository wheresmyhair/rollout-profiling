from typing import List, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class Job:
    id: int
    duration: float  # 预计完成时间
    assigned_worker: int = -1
    thread_id: int = -1  # 在worker的第几个线程上执行
    start_time: float = -1
    end_time: float = -1
    
    def __repr__(self):
        return f"Job({self.id}, {self.duration}s)"


@dataclass
class ThreadSlot:
    thread_id: int
    jobs: List[Job] = field(default_factory=list)
    
    def get_end_time(self) -> float:
        if not self.jobs:
            return 0.0
        return self.jobs[-1].end_time
    
    def get_available_time(self) -> float:
        return self.get_end_time()
    
    def add_job(self, job: Job):
        available_time = self.get_available_time()
        job.assigned_worker = -1
        job.thread_id = self.thread_id
        job.start_time = available_time
        job.end_time = available_time + job.duration
        self.jobs.append(job)


@dataclass
class Worker:
    id: int
    num_threads: int
    threads: List[ThreadSlot] = field(default_factory=list)
    
    def __post_init__(self):
        self.threads = [ThreadSlot(i) for i in range(self.num_threads)]
    
    def get_earliest_available_thread(self) -> ThreadSlot:
        return min(self.threads, key=lambda t: t.get_available_time())
    
    def get_total_end_time(self) -> float:
        return max([t.get_end_time() for t in self.threads], default=0.0)
    
    def assign_job(self, job: Job):
        thread = self.get_earliest_available_thread()
        job.assigned_worker = self.id
        thread.add_job(job)


class ThreadPoolScheduler:
    def __init__(self, num_workers: int, threads_per_worker: int, jobs: List[float]):
        self.num_workers = num_workers
        self.threads_per_worker = threads_per_worker
        self.jobs = [Job(i, duration) for i, duration in enumerate(jobs)]
        self.num_jobs = len(jobs)
    
    def greedy_lpt(self) -> Tuple[float, Dict]:
        sorted_jobs = sorted(self.jobs, key=lambda j: j.duration, reverse=True)
        
        workers = [Worker(i, self.threads_per_worker) for i in range(self.num_workers)]
        
        for job in sorted_jobs:
            best_worker = min(workers, 
                            key=lambda w: w.get_earliest_available_thread().get_available_time())
            best_worker.assign_job(job)
        
        total_time = max([w.get_total_end_time() for w in workers], default=0.0)
        
        return total_time, self._build_schedule(workers)
    
    def _build_schedule(self, workers: List[Worker]) -> Dict:
        schedule = {}
        for worker in workers:
            worker_schedule = {}
            for thread in worker.threads:
                worker_schedule[f"thread_{thread.thread_id}"] = [
                    {
                        'job_id': job.id,
                        'duration': job.duration,
                        'start_time': job.start_time,
                        'end_time': job.end_time
                    }
                    for job in thread.jobs
                ]
            schedule[f"worker_{worker.id}"] = worker_schedule
        return schedule
    
    
    def print_timeline(self, schedule: Dict, total_time: float):
        print("\n" + "=" * 80)
        print("Timeline view")
        print("=" * 80)
        
        time_points = set([0, total_time])
        for worker_name, worker_schedule in schedule.items():
            for thread_name, jobs in worker_schedule.items():
                for job in jobs:
                    time_points.add(job['start_time'])
                    time_points.add(job['end_time'])
        
        time_points = sorted(time_points)
        
        for worker_name in sorted(schedule.keys()):
            print(f"\n{worker_name}:")
            worker_schedule = schedule[worker_name]
            
            for thread_name in sorted(worker_schedule.keys()):
                jobs = worker_schedule[thread_name]
                timeline_str = ""
                
                current_time = 0
                for job in sorted(jobs, key=lambda x: x['start_time']):
                    if job['start_time'] > current_time:
                        gap = job['start_time'] - current_time
                        timeline_str += f"[Idle:{gap:.1f}s]"
                    
                    timeline_str += f"[J{job['job_id']}:{job['duration']:.1f}s]"
                    current_time = job['end_time']
                
                if current_time < total_time:
                    gap = total_time - current_time
                    timeline_str += f"[Idle:{gap:.1f}s]"
                
                print(f"  {thread_name}: {timeline_str}")
        
        print("\n" + "=" * 80)
    
    def print_summary(self, total_time: float, schedule: Dict):
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        
        print(f"\nProblem size:")
        print(f"  Worker: {self.num_workers}")
        print(f"  Threads per worker: {self.threads_per_worker}")
        print(f"  Jobs: {self.num_jobs}")
        print(f"  Total job time: {sum(j.duration for j in self.jobs):.2f}s")
        
        print(f"\nSchedule result:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Theoretical minimum: {sum(j.duration for j in self.jobs) / (self.num_workers * self.threads_per_worker):.2f}s")
        
        worker_times = []
        for worker_name in sorted(schedule.keys()):
            worker_schedule = schedule[worker_name]
            worker_time = max([max([j['end_time'] for j in jobs], default=0) 
                             for jobs in worker_schedule.values()], default=0)
            worker_times.append(worker_time)
        
        print(f"  Average worker completion time: {sum(worker_times) / len(worker_times):.2f}s")
        print(f"  Max worker completion time: {max(worker_times):.2f}s")
        print(f"  Min worker completion time: {min(worker_times):.2f}s")
        print(f"  Load balance index: {max(worker_times) / (sum(worker_times) / len(worker_times)):.2f}x")
        
        total_thread_time = sum(worker_times) * self.threads_per_worker
        total_job_time = sum(j.duration for j in self.jobs)
        utilization = (total_job_time / total_thread_time * 100) if total_thread_time > 0 else 0
        print(f"  Thread pool utilization: {utilization:.1f}%")
        
        print("\n" + "=" * 80)


def main():    
    x = 2
    y = 2
    jobs_1 = [5, 3, 8, 6, 2]
    
    print(f"\nWorker: {x}")
    print(f"Threads per worker: {y}")
    print(f"Jobs: {jobs_1}")
    
    scheduler_1 = ThreadPoolScheduler(x, y, jobs_1)
    total_time, schedule = scheduler_1.greedy_lpt()
    scheduler_1.print_timeline(schedule, total_time)
    scheduler_1.print_summary(total_time, schedule)
    
    x = 3
    y = 3
    jobs_2 = [10, 5, 8, 3, 6, 4, 7, 2, 9, 1]
    
    print(f"\nWorker: {x}")
    print(f"Threads per worker: {y}")
    print(f"Jobs: {jobs_2}")
    print(f"Total jobs: {len(jobs_2)}")
    
    scheduler_2 = ThreadPoolScheduler(x, y, jobs_2)
    total_time, schedule = scheduler_2.greedy_lpt()
    scheduler_2.print_timeline(schedule, total_time)
    scheduler_2.print_summary(total_time, schedule)



if __name__ == "__main__":
    main()
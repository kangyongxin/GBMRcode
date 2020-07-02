import threading

def job1():
    global A,lock
    lock.acquire()
    for i in range(7):
        A += 1
        print("JOB 1", A)
    lock.release()

def job2():
    global A,lock
    lock.acquire()
    for j in range(10):
        A = A*10
        print("JOB 2", A)
    lock.release()

def main():
    global A 
    A =0
    global lock
    lock = threading.Lock()
    
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()


if __name__=='__main__':
    main()
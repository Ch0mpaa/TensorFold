# Neutral control reference (for condition C)

Length-matched to the catalog (condition B). Content is deliberately
technical but has NO overlap with vulnerability taxonomy, CWE names,
CVE names, or security terminology. Every paragraph is factual
information about concurrency in the Python standard library, lifted
from the style of the CPython documentation. The purpose is to test
whether ANY long reference text improves the CWE answer (prompt
length / "longer-is-better") versus whether the catalog's CONTENT is
what helps. If C improves accuracy as much as B does, then whatever
helped was not the CWE definitions.

---

CPython-asyncio reference (neutral control):

Event loops are the heart of asyncio. They run asynchronous tasks and callbacks, perform network I/O operations, and run subprocesses. Application developers typically use high-level functions such as asyncio.run and should rarely need to reference the loop object.

Tasks are used to schedule coroutines concurrently. The asyncio.create_task function arranges for a coroutine to be executed and returns a Task object. The Task wraps the coroutine as a Future-compatible object that the event loop can schedule independently of the caller.

Streams are high-level primitives for working with network connections. The asyncio.open_connection coroutine returns a pair of StreamReader and StreamWriter objects. The reader provides coroutine methods such as read, readline, readexactly, and readuntil. The writer provides non-coroutine methods such as write, writelines, and close, and a coroutine wait_closed.

Transports and protocols are the low-level counterpart of streams. The transport abstracts a bidirectional byte stream and handles buffer management. The protocol receives callbacks from the transport when data arrives, the connection is made, or the connection is lost. Application code usually implements a protocol class and passes it to loop.create_connection or loop.create_server.

Synchronization primitives in asyncio mirror those in threading but are designed for single-threaded coroutine code. Lock provides mutual exclusion through acquire and release coroutines usable with async with. Event allows any number of coroutines to wait for a notification issued by set. Condition combines a lock with a wait queue that supports notify and notify_all. Semaphore and BoundedSemaphore implement counter-based admission control.

Queues let coroutines exchange data safely. Queue provides a FIFO; put and get are coroutines that block the calling task when full or empty. LifoQueue is a stack and PriorityQueue orders items by a comparison key. Each queue has a maxsize that acts as backpressure.

Subprocesses may be created from asyncio using create_subprocess_exec or create_subprocess_shell. The returned process object exposes stdin, stdout, and stderr as StreamWriter or StreamReader where appropriate, plus communicate for one-shot I/O. Signals delivered to the subprocess are the subprocess's own concern.

Concurrency across threads is supported by to_thread for CPU-light blocking calls and by run_in_executor for arbitrary executors. These helpers hand off work to a background thread and return a Future that resolves on the event loop when the work finishes.

Timers in asyncio are expressed as coroutines. The asyncio.sleep coroutine suspends the current task for a given delay, allowing other tasks to run. The loop.call_later and loop.call_at methods schedule a callback at a later time and return a Handle whose cancel method may be invoked before firing. Timeouts over a coroutine are expressed with asyncio.timeout as an async context manager that cancels the inner block if the deadline expires before completion.

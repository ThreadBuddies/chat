#pragma once

#include <coroutine>

namespace server {

/**
 * @brief A basic, fire-and-forget coroutine task type.
 * @note This is used as the return type for the internal lambda-coroutine
 *       in `await_suspend`. It allows us to launch a coroutine without
 *       needing to `co_await` its result or manage its lifetime.
 */
struct fire_and_forget_task {
    struct promise_type {
        static fire_and_forget_task get_return_object() noexcept { return {}; }
        static std::suspend_never initial_suspend() noexcept { return {}; }
        static std::suspend_never final_suspend() noexcept { return {}; }
        static void return_void() noexcept {}
        static void unhandled_exception() noexcept {
            // A production application should consider logging this event,
            // as an unhandled exception in a fire-and-forget task
            // can be difficult to debug.
        }
    };
};

/**
 * @brief An awaitable wrapper that ensures a coroutine resumes on a Drogon IO Loop.
 *
 * This class solves a common problem in asynchronous frameworks where a
 * `co_await` on a background operation (like a database query) would cause
 * the coroutine to resume on the background thread pool. This wrapper
 * intercepts the completion and posts the resumption back to the original
 * Drogon IO thread, preventing thread pool starvation and ensuring code
 * continues execution in the expected context.
 *
 * @tparam AwaiterType The type of the awaiter object to be wrapped. This must
 *         be a type that provides the awaiter interface (`await_ready`,
 *         `await_suspend`, `await_resume`).
 */
template <typename AwaiterType>
class resume_on_io_loop {
public:
    /**
     * @brief Constructs the wrapper, taking ownership of the inner awaiter.
     */
    explicit resume_on_io_loop(AwaiterType&& awaiter) noexcept
        : m_inner_awaiter(std::move(awaiter)) {}

    /**
     * @brief The co_await entry point for this wrapper.
     * @note This is rvalue-qualified (`&&`) because the wrapper itself is a
     *       temporary object that produces the final awaiter state machine.
     * @return An awaiter object that orchestrates the thread switching.
     */
    auto operator co_await() && noexcept {
        /**
         * @brief The internal state machine for the suspension and resumption process.
         */
        struct awaiter {
            /// The result type of the wrapped awaiter, with references and
            /// qualifiers removed to allow storage in a std::variant.
            using ResultType = std::decay_t<decltype(std::declval<AwaiterType&>().await_resume())>;

            /// The wrapped awaiter object, moved here for its lifetime management.
            AwaiterType m_inner_awaiter;
            /// The index of the IO thread where the coroutine was suspended.
            size_t m_original_thread_index;

            /// A variant to hold the outcome of the operation.
            /// `std::monostate` is the required initial state before completion.
            std::conditional_t<
                std::is_void_v<ResultType>,
                std::variant<std::monostate, std::exception_ptr>,
                std::variant<std::monostate, std::exception_ptr, ResultType>
            > m_result;

            /**
             * @brief Always returns false to force suspension for the thread switch.
             */
            bool await_ready() const noexcept {
                return false;
            }

            /**
             * @brief Called on the IO thread after resumption.
             * @return The value from the completed operation.
             * @throws The exception from the operation if it failed.
             */
            ResultType await_resume() {
                if(std::holds_alternative<std::exception_ptr>(m_result)) {
                    std::rethrow_exception(std::get<std::exception_ptr>(m_result));
                }
                if constexpr(!std::is_void_v<ResultType>) {
                    if(std::holds_alternative<ResultType>(m_result)) {
                        return std::get<ResultType>(std::move(m_result));
                    } else {
                        throw std::runtime_error("await_resume: neither exception nor value is in m_result");
                    }
                }
            }

            /**
             * @brief The core logic of the wrapper, executed upon suspension.
             */
            void await_suspend(std::coroutine_handle<> handle) {
                // On the original IO thread: capture the current context.
                m_original_thread_index = drogon::app().getCurrentThreadIndex();
                if(m_original_thread_index >= drogon::app().getThreadNum()) {
                    m_original_thread_index = 0; // Fallback to the first IO thread.
                }

                // Launch a fire-and-forget lambda-coroutine to perform the actual work.
                // It is safe to pass `this` because the `awaiter` object lives
                // in the frame of the suspended `handle`.
                // even tho this is captureless lambda and simply calling it as-is **should** be fine
                // lets get it's pointer with + to avoid any notion of potential ub
                auto lambda = +[](awaiter* self, std::coroutine_handle<> handle) -> fire_and_forget_task {
                    try {
                        // This co_await runs and completes on the background (e.g., DB) thread.
                        if constexpr(std::is_void_v<ResultType>) {
                            co_await self->m_inner_awaiter;
                        } else {
                            self->m_result.template emplace<ResultType>(std::move(co_await self->m_inner_awaiter));
                        }
                    } catch(...) {
                        // On failure, store the exception.
                        self->m_result.template emplace<std::exception_ptr>(std::current_exception());
                    }

                    // From the background thread, post the resumption back to the original IO thread.
                    drogon::app().getIOLoop(self->m_original_thread_index)->queueInLoop([handle]() {
                        handle.resume();
                    });
                };

                lambda(this, handle);
            }
        };

        return awaiter{std::move(m_inner_awaiter), 0, {}};
    }

private:
    AwaiterType m_inner_awaiter;
};

/**
 * @brief Helper function to create a resume_on_io_loop wrapper.
 *
 * This factory function allows for template argument deduction, providing a
 * clean and simple call site.
 *
 * @tparam AwaiterType The type of the awaiter to wrap, deduced automatically.
 * @param awaiter An rvalue-reference to the awaiter object to be wrapped.
 * @return A resume_on_io_loop object ready to be `co_await`ed.
 * @see resume_on_io_loop
 *
 * @code
 *   // Instead of this:
 *   // co_await drogon::orm::CoroMapper<...>(db).find(...);
 *
 *   // Use this to ensure resumption on the correct thread:
 *   co_await switch_to_io_loop(
 *       drogon::orm::CoroMapper<...>(db).find(...)
 *   );
 * @endcode
 */
template <typename AwaiterType>
auto switch_to_io_loop(AwaiterType&& awaiter) {
    return resume_on_io_loop<AwaiterType>(std::forward<AwaiterType>(awaiter));
}

} // namespace server

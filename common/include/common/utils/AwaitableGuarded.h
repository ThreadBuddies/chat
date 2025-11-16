#pragma once

#include <drogon/drogon.h>
#include <atomic>
#include <coroutine>
#include <memory>

namespace common {

/**
 * @class AwaitableGuarded
 * @brief A template class that bundles a data object with an async mutex.
 * @tparam T The type of the data to protect.
 *
 * @details This class provides a convenient and safe abstraction for managing a shared
 * resource in an cooperative asynchronous context. Instead of managing a mutex and data
 * separately, this class encapsulates them, significantly reducing the risk of
 * misuse.
 *
 * Access to the underlying data is exclusively provided through RAII-style proxy
 * objects. These proxies are obtained by `co_await`ing the `lock_shared()` or
 * `lock_unique()` methods. The lock is held for the lifetime of the proxy object,
 * and access to the data is provided via overloaded `operator->` and `operator*`.
 * This design ensures that the data can never be accessed without first acquiring
 * the appropriate lock.
 *
 * @note This class must be created via the static `create()` factory method and stored in shared_ptr.
 */
template <typename T>
class AwaitableGuarded : public std::enable_shared_from_this<AwaitableGuarded<T>> {
private:
    /// @brief The atomic state of the mutex. (0=free, -1=unique, >0=shared count)
    std::atomic<int> m_state{0};

    /// @brief The protected data
    T m_data;

    /// @brief Private tag to enforce creation via the `create()` factory method.
    struct private_tag {};

public:
    template <typename... Args>
    explicit AwaitableGuarded(private_tag, Args&&... args) : m_data{std::forward<Args>(args)...} {}

    /**
     * @brief Factory method to create a new lifetime-safe AwaitableGuarded instance.
     * @param args Arguments to be forwarded to T's constructor.
     * @return A `std::shared_ptr<AwaitableGuarded<T>>` managing the new instance.
     */
    template <typename... Args>
    static auto create(Args&&... args) {
        // use make_shared with a tag instead of shared_ptr<T>(new T(...)) to avoid extra malloc
        return std::make_shared<AwaitableGuarded<T>>(private_tag{}, std::forward<Args>(args)...);
    }

    /**
     * @class SharedProxy
     * @brief A proxy object providing temporary, read-only access to the guarded data.
     */
    class SharedProxy {
    public:
        ~SharedProxy() {
            if(m_guarded) {
                m_guarded->m_state.fetch_sub(1, std::memory_order_release);
            }
        }

        SharedProxy(const SharedProxy&) = delete;
        SharedProxy& operator=(const SharedProxy&) = delete;
        SharedProxy(SharedProxy&&) noexcept = default;
        SharedProxy& operator=(SharedProxy&&) = delete;

        [[nodiscard]] const T* operator->() const noexcept { return &m_guarded->m_data; }
        [[nodiscard]] const T& operator*() const noexcept { return m_guarded->m_data; }

    private:
        friend class AwaitableGuarded<T>;
        explicit SharedProxy(std::shared_ptr<AwaitableGuarded<T>> guarded) noexcept 
            : m_guarded{std::move(guarded)} {}

        const std::shared_ptr<AwaitableGuarded<T>> m_guarded;
    };

    /**
     * @class UniqueProxy
     * @brief A proxy object providing temporary, read-write access to the guarded data.
     */
    class UniqueProxy {
    public:
        ~UniqueProxy() {
            if(m_guarded) {
                m_guarded->m_state.store(0, std::memory_order_release);
            }
        }

        UniqueProxy(const UniqueProxy&) = delete;
        UniqueProxy& operator=(const UniqueProxy&) = delete;
        UniqueProxy(UniqueProxy&&) noexcept = default;
        UniqueProxy& operator=(UniqueProxy&&) = delete;

        [[nodiscard]] T* operator->() noexcept { return &m_guarded->m_data; }
        [[nodiscard]] T& operator*() noexcept { return m_guarded->m_data; }

    private:
        friend class AwaitableGuarded<T>;
        explicit UniqueProxy(std::shared_ptr<AwaitableGuarded<T>> guarded) noexcept 
            : m_guarded{std::move(guarded)} {}

        const std::shared_ptr<AwaitableGuarded<T>> m_guarded;
    };

private:
    /**
     * @brief A base template for lock awaitables to handle common async machinery.
     * @tparam Derived The concrete awaitable class.
     * @tparam ProxyType The proxy type to be returned on resumption.
     */
    template <typename Derived, typename ProxyType>
    class LockAwaitableBase {
    protected:
        std::shared_ptr<AwaitableGuarded<T>> guarded;
        std::coroutine_handle<> handle_ = nullptr;

    public:
        bool await_ready() noexcept {
            return static_cast<Derived*>(this)->try_lock();
        }

        void await_suspend(std::coroutine_handle<> h) noexcept {
            handle_ = h;
            drogon::app().getIOLoop(drogon::app().getCurrentThreadIndex())->queueInLoop([this] { 
                spin(); 
            });
        }

        ProxyType await_resume() noexcept {
            return ProxyType{std::move(guarded)};
        }

    private:
        void spin() {
            if(static_cast<Derived*>(this)->try_lock()) {
                // Post resumption to the event loop to keep the call stack shallow.
                drogon::app().getIOLoop(drogon::app().getCurrentThreadIndex())->queueInLoop([h = handle_] { h.resume(); });
            } else {
                // Re-queue this task to try again later, this gurantees forward progress.
                drogon::app().getIOLoop(drogon::app().getCurrentThreadIndex())->queueInLoop([this] {
                    spin();
                });
            }
        }
    };

    /**
     * @brief An awaitable object for acquiring a shared lock.
     */
    class SharedLockAwaitable : public LockAwaitableBase<SharedLockAwaitable, SharedProxy> {
    public:
        explicit SharedLockAwaitable(std::shared_ptr<AwaitableGuarded<T>> g) {
            this->guarded = std::move(g);
        }

    private:
        friend class LockAwaitableBase<SharedLockAwaitable, SharedProxy>;

        [[nodiscard]] bool try_lock() noexcept {
            auto current = this->guarded->m_state.load(std::memory_order_acquire);
            if(current >= 0) {
                return this->guarded->m_state.compare_exchange_strong(current, current + 1, std::memory_order_acq_rel);
            }
            return false;
        }
    };

    /**
     * @brief An awaitable object for acquiring a unique lock.
     */
    class UniqueLockAwaitable : public LockAwaitableBase<UniqueLockAwaitable, UniqueProxy> {
    public:
        explicit UniqueLockAwaitable(std::shared_ptr<AwaitableGuarded<T>> g) {
            this->guarded = std::move(g);
        }

    private:
        friend class LockAwaitableBase<UniqueLockAwaitable, UniqueProxy>;

        [[nodiscard]] bool try_lock() noexcept {
            int expected = 0;
            return this->guarded->m_state.compare_exchange_strong(expected, -1, std::memory_order_acq_rel);
        }
    };

public:
    /**
     * @brief Creates an awaitable to acquire a shared (reader) lock.
     * @return A `SharedLockAwaitable` object to be used with `co_await`.
     */
    [[nodiscard]] SharedLockAwaitable lock_shared() {
        return SharedLockAwaitable{this->shared_from_this()};
    }

    /**
     * @brief Creates an awaitable to acquire a unique (reader-writer) lock.
     * @return A `UniqueLockAwaitable` object to be used with `co_await`.
     */
    [[nodiscard]] UniqueLockAwaitable lock_unique() {
        return UniqueLockAwaitable{this->shared_from_this()};
    }

    /**
     * @brief Verifies if this AwaitableGuarded object is the container for the given data reference.
     *
     * This method enables safe re-entrant patterns by allowing functions to check
     * if a provided data reference corresponds to a specific guarded object,
     * thus avoiding attempts to re-lock a mutex already held by the caller.
     *
     * @param a_data A const reference to a raw data object to check.
     * @return `true` if this instance holds the provided data object, `false` otherwise.
     */
    [[nodiscard]] bool isHolding(const T& a_data) const noexcept {
        // Compare memory addresses to check if the provided data is the one we are guarding.
        return std::addressof(this->m_data) == std::addressof(a_data);
    }
};

} // namespace common

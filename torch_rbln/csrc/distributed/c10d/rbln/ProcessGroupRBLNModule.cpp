#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/extension.h>
#include <torch_rbln/csrc/distributed/c10d/rbln/ProcessGroupRBLN.hpp>
#include <torch_rbln/csrc/distributed/c10d/rbln/ProcessGroupRBLNModule.hpp>
#include <array>

namespace {

/**
 * @brief Wrapper to ensure GIL is released before destructing ProcessGroupRBLN
 *
 * This wrapper class ensures that the Global Interpreter Lock (GIL) is properly
 * released before destructing ProcessGroupRBLN instances, preventing potential
 * deadlocks in multi-threaded environments.
 */
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_{};

 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) noexcept = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor& operator=(IntrusivePtrNoGilDestructor&&) noexcept = default;

  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl) : impl_(std::move(impl)) {}

  explicit IntrusivePtrNoGilDestructor(T* impl) : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}

  ~IntrusivePtrNoGilDestructor() { // NOLINT
    if (impl_) {
      if (PyGILState_Check() != 0) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }

  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  [[nodiscard]] T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return impl_;
  }
};

} // anonymous namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true) // NOLINT

template <typename T>
using intrusive_ptr_no_gil_destructor_class_ = py::class_<T, IntrusivePtrNoGilDestructor<T>>;

namespace torch_rbln::distributed {

PyObject* initialize_process_group_rbln_bindings(PyObject* _unused, PyObject* noargs) { // NOLINT
  // Import the main torch_rbln._C module
  auto torch_rbln_module = THPObjectPtr(PyImport_ImportModule("torch_rbln._C"));
  if (!torch_rbln_module) {
    throw python_error();
  }
  auto torch_rbln_module_handle = py::handle(torch_rbln_module).cast<py::module>();

  // Check if already initialized to prevent double initialization
  if (py::hasattr(torch_rbln_module_handle, "_distributed_c10d")) {
    Py_RETURN_TRUE;
  }

  // Create the _distributed_c10d submodule
  auto distributed_submodule = torch_rbln_module_handle.def_submodule("_distributed_c10d", "distributed c10d bindings");
  auto distributed_module_handle = py::handle(distributed_submodule).cast<py::module>();

  // Default timeout for process groups (30 minutes)
  constexpr auto kDefaultProcessGroupTimeout = std::chrono::milliseconds(30 * 60 * 1000);

  // Import torch._C._distributed_c10d to get the Backend class for inheritance
  py::module_ torch_distributed_module = py::module_::import("torch._C._distributed_c10d");
  auto backend_base_class = torch_distributed_module.attr("Backend");

  // Bind ProcessGroupRBLN class with proper inheritance from Backend
  auto process_group_rbln_class = intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupRBLN>(
      distributed_module_handle, "ProcessGroupRBLN", backend_base_class);

  // Constructor 1: With explicit options
  process_group_rbln_class
      .def(
          py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                      int rank,
                      int size,
                      int group_id,
                      const std::vector<int>& global_ranks_in_group,
                      const c10::intrusive_ptr<::c10d::Backend::Options>& options,
                      const py::object& gloo_backend) {
            c10::intrusive_ptr<::c10d::Backend> gloo_ptr;
            if (!gloo_backend.is_none()) {
              gloo_ptr = gloo_backend.cast<c10::intrusive_ptr<::c10d::Backend>>();
            }
            py::gil_scoped_release gil_release{};
            return c10::make_intrusive<::c10d::ProcessGroupRBLN>(
                store, rank, size, group_id, global_ranks_in_group, options, std::move(gloo_ptr));
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("group_id"),
          py::arg("global_ranks_in_group"),
          py::arg("options"),
          py::arg("gloo_backend") = py::none(),
          R"(Create a new ProcessGroupRBLN instance with explicit options. Pass gloo_backend for non-float16 allreduce/reduce_scatter fallback.)")

      // Constructor 2: With timeout parameter (supports int, timedelta, or None)
      .def(
          py::init([kDefaultProcessGroupTimeout](
                       const c10::intrusive_ptr<::c10d::Store>& store,
                       int rank,
                       int size,
                       int group_id,
                       const std::vector<int>& global_ranks_in_group,
                       const py::object& timeout_object,
                       const py::object& gloo_backend) {
            // Parse timeout parameter - supports None, int (milliseconds), or timedelta
            std::chrono::milliseconds timeout_duration;
            if (timeout_object.is_none()) {
              timeout_duration = kDefaultProcessGroupTimeout;
            } else if (py::isinstance<py::int_>(timeout_object)) {
              timeout_duration = std::chrono::milliseconds(timeout_object.cast<int>());
            } else {
              // Handle datetime.timedelta objects
              auto total_seconds = timeout_object.attr("total_seconds")().cast<double>();
              timeout_duration = std::chrono::milliseconds(static_cast<int>(total_seconds * 1000));
            }

            c10::intrusive_ptr<::c10d::Backend> gloo_ptr;
            if (!gloo_backend.is_none()) {
              gloo_ptr = gloo_backend.cast<c10::intrusive_ptr<::c10d::Backend>>();
            }
            auto backend_options = c10::make_intrusive<::c10d::Backend::Options>("rbln", timeout_duration);
            py::gil_scoped_release gil_release{};
            return c10::make_intrusive<::c10d::ProcessGroupRBLN>(
                store, rank, size, group_id, global_ranks_in_group, backend_options, std::move(gloo_ptr));
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("group_id"),
          py::arg("global_ranks_in_group"),
          py::arg("timeout") = py::none(),
          py::arg("gloo_backend") = py::none(),
          R"(Create a new ProcessGroupRBLN instance with timeout. Pass gloo_backend for non-float16 allreduce/reduce_scatter fallback.)")

      // Expose key methods
      .def("get_rank", &::c10d::ProcessGroupRBLN::getRank, "Get the rank of this process group")
      .def("get_size", &::c10d::ProcessGroupRBLN::getSize, "Get the size of this process group")
      .def("get_backend_name", &::c10d::ProcessGroupRBLN::getBackendName, "Get the backend name")

      // Sequence number management methods
      .def(
          "_set_sequence_number_for_group",
          &::c10d::ProcessGroupRBLN::setSequenceNumberForGroup,
          py::call_guard<py::gil_scoped_release>(),
          "Set sequence number for the group")
      .def(
          "_get_sequence_number_for_group",
          &::c10d::ProcessGroupRBLN::getSequenceNumberForGroup,
          py::call_guard<py::gil_scoped_release>(),
          "Get the current sequence number for the group");

  Py_RETURN_TRUE;
}

// Method definitions for the distributed module
static std::array<PyMethodDef, 2> distributed_method_definitions = {
    {{"_c10d_rbln_init", initialize_process_group_rbln_bindings, METH_NOARGS, nullptr},
     {nullptr, nullptr, 0, nullptr}}};

PyMethodDef* get_distributed_method_definitions() { // NOLINT
  return distributed_method_definitions.data();
}

} // namespace torch_rbln::distributed

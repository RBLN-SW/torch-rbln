#ifndef REBEL_COMMON_STATUS_H
#define REBEL_COMMON_STATUS_H

#include <rebel/common/utility.h>

#include <memory>
#include <string>

namespace rbln {

#define RT_RETURN_IF_ERROR(expr) \
  do {                           \
    auto _status = (expr);       \
    if ((!_status.IsOK())) {     \
      return _status;            \
    }                            \
  } while (0)

#define RT_CHECK(cond, code, debug_msg)                                    \
  do {                                                                     \
    bool _cond = (cond);                                                   \
    if (!_cond) {                                                          \
      return ::rbln::Status(::rbln::StatusCode::code).DebugMsg(debug_msg); \
    }                                                                      \
  } while (0)

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define RT_CHECK_DBG(cond, code) RT_CHECK(cond, code, #cond " at " __FILE__ ":" TOSTRING(__LINE__))

// StatusCode represents error messages to be displayed to the users. The messages specified with
// `ExternMsg` function will be displayed with the error description in `statuscode_strings` defined
// in status.cc.
enum class StatusCode : int {
  kOk = 0,
  kInvalidArgument = 50,
  kLoadingFailed = 101,
  kLoadingFileNotFound = 102,
  kLoadingInvalidFile = 103,
  kLoadingUnexpectedEof = 104,
  kLoadingInvalidVersion = 105,
  kLoadingInvalidFileContent = 106,
  kLoadingInvalidChecksum = 107,
  kLoadingInvalidSharedObj = 108,
  kInitInternal = 201,
  kInitInvalidArgument = 202,
  kInitAlreadyCreated = 203,
  kInitFailedLoadingSharedObj = 204,
  kInitSharedObjNoSymbol = 205,
  kInitMemAllocFailed = 206,
  kRunInternal = 301,
  kRunErrorFromOtherThread = 302,
  kSysError = 501,
  kSysWaitJobBusy = 502,
  kSysWaitJobKernelTimeout = 503,
  kSysWaitJobTaskAborted = 504,
  kSysNOENT = 505,
  kSysSRCH = 506,
  kSysINTR = 507,
  kSysIO = 508,
  kSysNXIO = 509,
  kSysNOMEM = 510,
  kSysBUSY = 511,
  kSysNODEV = 512,
  kSysNOSPC = 513,
  kSysPIPE = 514,
  kSysCANCELLED = 515,
  kCompileInternal = 601,
  kCompileInvalidArgument = 602,
  kRcclGetUniqueIdFailed = 701,
  kRcclCommInitRankFailed = 702,
  kRcclAddressInUse = 703,
  kRcclCommSplitFailed = 704,
  kLibraryLoadFailed = 801,
};

class [[nodiscard]] Status final {
 public:
  Status() noexcept = default;
  explicit Status(StatusCode code);
  Status(StatusCode code, int sys_code);

  Status(Status&&) = default;
  Status& operator=(Status&&) = default;
  ~Status() = default;

  Status(const Status& other)
      : state_((other.state_ == nullptr) ? nullptr : new State(*other.state_)) {}

  Status& operator=(const Status& other) {
    if (state_ != other.state_) {
      if (other.state_ == nullptr) {
        state_.reset();
      } else {
        state_.reset(new State(*other.state_));
      }
    }
    return *this;
  }

  bool IsOK() const { return (state_ == nullptr); }

  int Code() const noexcept;

  std::string ErrorMessage() const noexcept;

  std::string ToProdString() const;

  std::string ToDebugString() const;

  inline Status ExternMsg(const std::string& user_msg) {
    if (state_ != nullptr) {
      state_->extern_msg_ = user_msg;
    }
    return *this;
  }

  inline Status DebugMsg(const std::string& debug_msg) {
#ifndef REBEL_DEPLOY
    if (state_ != nullptr) {
      state_->debug_msg_ = debug_msg;
    }
#endif
    return *this;
  }

  bool operator==(const Status& other) const {
    if (state_ == nullptr) {
      return other.state_ == nullptr;
    }
    return state_->code_ == other.state_->code_ && state_->sys_code_ == other.state_->sys_code_ &&
           state_->extern_msg_ == other.state_->extern_msg_ &&
           state_->debug_msg_ == other.state_->debug_msg_;
  }

  bool operator!=(const Status& other) const { return !(*this == other); }

  inline void IgnoreError() const {
    // no-op
  }

 private:
  struct State {
    State(StatusCode code, int sys_code, const std::string& extern_msg)
        : code_(code), sys_code_(sys_code), extern_msg_(extern_msg) {}

    const StatusCode code_;
    const int sys_code_;
    std::string extern_msg_;
    std::string debug_msg_;
  };
  // As long as Code() is OK, state_ == nullptr.
  std::unique_ptr<State> state_;
};

Status OkStatus();
Status InitInternalError();
Status InitMemAllocFailedError();
Status InitInvalidArgumentError();
Status InitAlreadyCreatedError();
Status RunInternalError();
Status SysError(int sys_code);
Status CleanupInternalError();
Status CompileInternalError();
Status CompileInvalidArgumentError();
Status RcclGetUniqueIdFailed();
Status RcclCommInitRankFailed();
Status RcclAddressInUse();
Status RcclCommSplitFailed();
Status LibraryLoadError();

}  // namespace rbln
#endif  // REBEL_COMMON_STATUS_H

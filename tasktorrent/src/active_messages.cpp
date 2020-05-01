#ifndef TTOR_SHARED

#include "active_messages.hpp"

namespace ttor {

int ActiveMsgBase::get_id() const { return id_; }
bool ActiveMsgBase::is_bound_to_MPI_master() const { return bound_; }
void ActiveMsgBase::allow_on_worker() {
    bound_ = false;
}
ActiveMsgBase::ActiveMsgBase(int id) : id_(id), bound_(true) {}
ActiveMsgBase::~ActiveMsgBase(){}

}

#endif
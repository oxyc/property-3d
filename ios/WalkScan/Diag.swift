import Foundation
#if os(iOS)
import os
#endif

/// Process memory, for diag.jsonl and for shedding load before iOS kills the app.
enum Memory {
    /// `phys_footprint`: the number iOS compares against the per-app limit (jetsam).
    static func footprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
    }

    /// Bytes the app can still allocate before it is killed (0 where the OS cannot say).
    static func availableBytes() -> Int {
        #if os(iOS)
        return os_proc_available_memory()
        #else
        return 0
        #endif
    }

    /// Shed load below this much headroom.
    static let lowAvailable = 500_000_000
}

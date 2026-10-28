#include <fcntl.h>
#include <linux/kvm.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define MEM_SIZE (4u << 20)

static int kvm, vm, vcpu;
static uint8_t *mem;
static struct kvm_run *run;

static void die(const char *what) {
    printf("KVMTEST FAIL: %s\n", what);
    fflush(stdout);
    _exit(1);
}

static void setup(const uint8_t *code, size_t length, uint64_t load) {
    memset(mem, 0, MEM_SIZE);
    memcpy(mem + load, code, length);
}

static void flat_segment(struct kvm_segment *s, uint16_t selector, int code, int is32, int is64) {
    memset(s, 0, sizeof *s);
    s->base = 0;
    s->limit = is32 || is64 ? 0xffffffff : 0xffff;
    s->selector = selector;
    s->type = code ? 11 : 3;
    s->present = 1;
    s->dpl = 0;
    s->db = is32 && !is64;
    s->s = 1;
    s->l = code && is64;
    s->g = is32 || is64;
}

/* Runs until HLT; returns the bytes written to port 0x3f8. */
static int run_guest(char *out, size_t cap) {
    size_t n = 0;
    for (int i = 0; i < 100; i++) {
        if (ioctl(vcpu, KVM_RUN, 0) < 0) die("KVM_RUN");
        switch (run->exit_reason) {
        case KVM_EXIT_IO:
            if (run->io.direction == KVM_EXIT_IO_OUT && run->io.port == 0x3f8) {
                uint8_t c = *((uint8_t *)run + run->io.data_offset);
                if (n + 1 < cap) out[n++] = c;
            }
            break;
        case KVM_EXIT_HLT:
            out[n] = 0;
            return 0;
        default:
            printf("KVMTEST: unexpected exit %d (hw=%llx)\n", run->exit_reason,
                   (unsigned long long)run->hw.hardware_exit_reason);
            out[n] = 0;
            return -1;
        }
    }
    out[n] = 0;
    return -2;
}

static void reset_vcpu(void) {
    struct kvm_regs regs;
    memset(&regs, 0, sizeof regs);
    regs.rflags = 2;
    if (ioctl(vcpu, KVM_SET_REGS, &regs) < 0) die("SET_REGS");
}

static int test(const char *name, const char *expect, void (*prepare)(void), const uint8_t *code, size_t length) {
    char out[64];
    setup(code, length, 0x1000);
    prepare();
    int status = run_guest(out, sizeof out);
    int ok = status == 0 && strcmp(out, expect) == 0;
    printf("KVMTEST %s: %s (got \"%s\" expected \"%s\" status %d)\n", name, ok ? "OK" : "FAIL", out, expect, status);
    fflush(stdout);
    return ok;
}

static void prepare_real(void) {
    struct kvm_sregs sregs;
    if (ioctl(vcpu, KVM_GET_SREGS, &sregs) < 0) die("GET_SREGS");
    sregs.cs.base = 0;
    sregs.cs.selector = 0;
    sregs.cr0 &= ~1ull;
    if (ioctl(vcpu, KVM_SET_SREGS, &sregs) < 0) die("SET_SREGS");
    reset_vcpu();
    struct kvm_regs regs;
    ioctl(vcpu, KVM_GET_REGS, &regs);
    regs.rip = 0x1000;
    regs.rflags = 2;
    ioctl(vcpu, KVM_SET_REGS, &regs);
}

static void prepare_protected(void) {
    struct kvm_sregs sregs;
    ioctl(vcpu, KVM_GET_SREGS, &sregs);
    sregs.cr0 = 0x11; /* PE | ET */
    sregs.cr4 = 0;
    sregs.efer = 0;
    flat_segment(&sregs.cs, 8, 1, 1, 0);
    flat_segment(&sregs.ds, 16, 0, 1, 0);
    sregs.es = sregs.fs = sregs.gs = sregs.ss = sregs.ds;
    if (ioctl(vcpu, KVM_SET_SREGS, &sregs) < 0) die("SET_SREGS prot");
    struct kvm_regs regs;
    memset(&regs, 0, sizeof regs);
    regs.rip = 0x1000;
    regs.rsp = 0x8000;
    regs.rflags = 2;
    ioctl(vcpu, KVM_SET_REGS, &regs);
}

static void identity_tables(int pae_only) {
    uint64_t *pml4 = (uint64_t *)(mem + 0x2000);
    uint64_t *pdpt = (uint64_t *)(mem + 0x3000);
    uint64_t *pd = (uint64_t *)(mem + 0x4000);
    if (pae_only) {
        /* 3 entries PDPT at 0x2000 (32-byte aligned), PD at 0x3000 */
        uint64_t *pdp = (uint64_t *)(mem + 0x2000);
        uint64_t *dir = (uint64_t *)(mem + 0x3000);
        pdp[0] = 0x3000 | 1;
        dir[0] = 0x0 | 0x83;
        dir[1] = 0x200000 | 0x83;
        return;
    }
    pml4[0] = 0x3000 | 3;
    pdpt[0] = 0x4000 | 3;
    pd[0] = 0x0 | 0x83;
    pd[1] = 0x200000 | 0x83;
}

static void prepare_long(void) {
    identity_tables(0);
    struct kvm_sregs sregs;
    ioctl(vcpu, KVM_GET_SREGS, &sregs);
    sregs.cr0 = 0x80000011;
    sregs.cr3 = 0x2000;
    sregs.cr4 = 0x20;
    sregs.efer = 0x500;
    flat_segment(&sregs.cs, 8, 1, 1, 1);
    flat_segment(&sregs.ds, 16, 0, 1, 1);
    sregs.es = sregs.fs = sregs.gs = sregs.ss = sregs.ds;
    if (ioctl(vcpu, KVM_SET_SREGS, &sregs) < 0) die("SET_SREGS long");
    struct kvm_regs regs;
    memset(&regs, 0, sizeof regs);
    regs.rip = 0x1000;
    regs.rsp = 0x8000;
    regs.rflags = 2;
    ioctl(vcpu, KVM_SET_REGS, &regs);
}

static void prepare_pae(void) {
    identity_tables(1);
    struct kvm_sregs sregs;
    ioctl(vcpu, KVM_GET_SREGS, &sregs);
    sregs.cr0 = 0x80000011;
    sregs.cr3 = 0x2000;
    sregs.cr4 = 0x20;
    sregs.efer = 0;
    flat_segment(&sregs.cs, 8, 1, 1, 0);
    flat_segment(&sregs.ds, 16, 0, 1, 0);
    sregs.es = sregs.fs = sregs.gs = sregs.ss = sregs.ds;
    if (ioctl(vcpu, KVM_SET_SREGS, &sregs) < 0) die("SET_SREGS pae");
    struct kvm_regs regs;
    memset(&regs, 0, sizeof regs);
    regs.rip = 0x1000;
    regs.rsp = 0x8000;
    regs.rflags = 2;
    ioctl(vcpu, KVM_SET_REGS, &regs);
}

static void prepare_legacy_paging(void) {
    uint32_t *directory = (uint32_t *)(mem + 0x2000);
    directory[0] = 0x0 | 0x83; /* 4 MiB page, needs PSE */
    struct kvm_sregs sregs;
    ioctl(vcpu, KVM_GET_SREGS, &sregs);
    sregs.cr0 = 0x80000011;
    sregs.cr3 = 0x2000;
    sregs.cr4 = 0x10; /* PSE */
    sregs.efer = 0;
    flat_segment(&sregs.cs, 8, 1, 1, 0);
    flat_segment(&sregs.ds, 16, 0, 1, 0);
    sregs.es = sregs.fs = sregs.gs = sregs.ss = sregs.ds;
    if (ioctl(vcpu, KVM_SET_SREGS, &sregs) < 0) die("SET_SREGS legacy paging");
    struct kvm_regs regs;
    memset(&regs, 0, sizeof regs);
    regs.rip = 0x1000;
    regs.rsp = 0x8000;
    regs.rflags = 2;
    ioctl(vcpu, KVM_SET_REGS, &regs);
}

/* mov dx, 0x3f8; mov al, 'R'; out dx, al; mov al, 'M'; out dx, al; hlt */
static const uint8_t real_code[] = {0xba, 0xf8, 0x03, 0xb0, 'R', 0xee, 0xb0, 'M', 0xee, 0xf4};
/* mov dx,0x3f8; mov al,'P'; out dx,al; mov eax,0x12345678; push eax; pop ebx; cmp ebx,eax; jne +4; mov al,'K'; out dx,al; hlt */
static const uint8_t prot_code[] = {0xba, 0xf8, 0x03, 0, 0, 0xb0, 'P', 0xee, 0xb8, 0x78, 0x56, 0x34, 0x12,
                                    0x50, 0x5b, 0x39, 0xc3, 0x75, 0x03, 0xb0, 'K', 0xee, 0xf4};
/* mov edx,0x3f8; mov al,'L'; out dx,al; mov rax,0x1122334455667788; mov rcx,rax; shr rcx,56; cmp cl,0x11; jne +3; mov al,'K'; out dx,al; hlt */
static const uint8_t long_code[] = {0xba, 0xf8, 0x03, 0, 0, 0xb0, 'L', 0xee, 0x48, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44,
                                    0x33, 0x22, 0x11, 0x48, 0x89, 0xc1, 0x48, 0xc1, 0xe9, 0x38, 0x80, 0xf9, 0x11,
                                    0x75, 0x03, 0xb0, 'K', 0xee, 0xf4};

int main(void) {
    printf("KVMTEST start\n");
    fflush(stdout);
    kvm = open("/dev/kvm", O_RDWR);
    if (kvm < 0) die("open /dev/kvm");
    int version = ioctl(kvm, KVM_GET_API_VERSION, 0);
    if (version != 12) die("API version");
    vm = ioctl(kvm, KVM_CREATE_VM, 0);
    if (vm < 0) die("CREATE_VM");
    mem = mmap(0, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) die("mmap");
    struct kvm_userspace_memory_region region = {.slot = 0, .guest_phys_addr = 0, .memory_size = MEM_SIZE,
                                                  .userspace_addr = (uint64_t)mem};
    if (ioctl(vm, KVM_SET_USER_MEMORY_REGION, &region) < 0) die("SET_USER_MEMORY_REGION");
    vcpu = ioctl(vm, KVM_CREATE_VCPU, 0);
    if (vcpu < 0) die("CREATE_VCPU");
    struct kvm_cpuid2 *cpuid = calloc(1, sizeof *cpuid + 256 * sizeof(struct kvm_cpuid_entry2));
    cpuid->nent = 256;
    if (ioctl(kvm, KVM_GET_SUPPORTED_CPUID, cpuid) < 0) die("GET_SUPPORTED_CPUID");
    printf("KVMTEST supported cpuid entries: %u\n", cpuid->nent);
    if (ioctl(vcpu, KVM_SET_CPUID2, cpuid) < 0) die("SET_CPUID2");
    int size = ioctl(kvm, KVM_GET_VCPU_MMAP_SIZE, 0);
    run = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, vcpu, 0);
    if (run == MAP_FAILED) die("mmap run");
    int failures = 0;
    failures += !test("real-mode", "RM", prepare_real, real_code, sizeof real_code);
    failures += !test("protected-32", "PK", prepare_protected, prot_code, sizeof prot_code);
    failures += !test("long-mode", "LK", prepare_long, long_code, sizeof long_code);
    failures += !test("pae-paging", "PK", prepare_pae, prot_code, sizeof prot_code);
    failures += !test("32-bit-paging", "PK", prepare_legacy_paging, prot_code, sizeof prot_code);
    printf("KVMTEST done failures=%d\n", failures);
    fflush(stdout);
    return failures != 0;
}

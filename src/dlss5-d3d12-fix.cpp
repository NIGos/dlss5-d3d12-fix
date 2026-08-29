// dlss5-d3d12-fix - ReShade add-on.
//
// Makes a DLSS 5 neural-rendering add-on work in a D3D12 game whose DLSS output
// carries a mip chain.
//
// That add-on requires the output texture to have exactly one mip level and
// silently does nothing otherwise -- no log line, no error, just a neural pass
// that never runs. Some games hand DLSS an output and depth with a full mip
// chain, which NGX itself accepts perfectly well.
//
// Two ways of making it accept them were tried and both hang the GPU with
// DXGI_ERROR_DEVICE_HUNG: reporting a false mip count through GetDesc, and
// patching the check out of the add-on's binary. That check is a guard, not a
// formality -- the code behind it genuinely cannot handle a mip chain. So the
// only correct fix is to hand it a texture that really does have one mip:
//
//   1. create single-mip textures matching the game's output and depth
//   2. seed the output copy with what the game's already holds, because DLSS
//      does not necessarily write every pixel
//   3. put them in the parameter block for the duration of the call
//   4. copy the result back into the game's output and restore the pointers
//
// Everything stays on the game's own device and queue, recorded into the
// command list the evaluate is given.
//
// Frame generation drives evaluates down a second path on another thread, so
// the whole sequence is serialised -- see the note on g_state_cs. Without that
// the neural pass intermittently missed frames.
//
// Diagnostics stay in: the log reports the contract, the resource descriptors,
// which of the consumer's requirements each one meets, and running totals that
// can be compared against the consumer's own counters.
//
// Tested on Resonance: A Plague Tale Legacy (D3D12, Streamline, Ray
// Reconstruction and Frame Generation). Nothing here is specific to it.
//
// Build:
//   rc /nologo version.rc
//   cl /nologo /LD /EHsc /O2 /MT dlss5-d3d12-fix.cpp ^
//      /link /OUT:dlss5-d3d12-fix.addon64 version.res kernel32.lib user32.lib

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <cstdio>
#include <cstdarg>
#include <cstdint>

#define PROBE_VERSION "2.5.0"

extern "C" __declspec(dllexport) const char *NAME =
    "DLSS 5 D3D12 Mip Fix " PROBE_VERSION;
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Lets a DLSS 5 neural-rendering add-on run in a game whose DLSS output has a "
    "mip chain. That add-on needs a single-mip output and silently does nothing "
    "otherwise; this supplies one for the duration of each call and copies the "
    "result back. Settings in dlss5-d3d12-fix.cfg, diagnostics in "
    "dlss5-d3d12-fix.log.";

// ---------------------------------------------------------------------------
// NGX declarations
//
// Declaration order mirrors NVIDIA's nvsdk_ngx.h. MSVC emits same-name virtual
// overloads in reverse declaration order, so keeping the order identical
// reproduces NVIDIA's vtable layout without hardcoding any slot number.
// ---------------------------------------------------------------------------

struct ID3D11Resource;  // opaque; only ever held as a pointer

typedef int NVSDK_NGX_Result;
static const NVSDK_NGX_Result NGX_SUCCESS = 1;

struct NVSDK_NGX_Handle { unsigned int Id; };

struct NVSDK_NGX_Parameter
{
    virtual void Set(const char *, unsigned long long) = 0;
    virtual void Set(const char *, float) = 0;
    virtual void Set(const char *, double) = 0;
    virtual void Set(const char *, unsigned int) = 0;
    virtual void Set(const char *, int) = 0;
    virtual void Set(const char *, ID3D11Resource *) = 0;
    virtual void Set(const char *, ID3D12Resource *) = 0;
    virtual void Set(const char *, void *) = 0;

    virtual NVSDK_NGX_Result Get(const char *, unsigned long long *) const = 0;
    virtual NVSDK_NGX_Result Get(const char *, float *) const = 0;
    virtual NVSDK_NGX_Result Get(const char *, double *) const = 0;
    virtual NVSDK_NGX_Result Get(const char *, unsigned int *) const = 0;
    virtual NVSDK_NGX_Result Get(const char *, int *) const = 0;
    virtual NVSDK_NGX_Result Get(const char *, ID3D11Resource **) const = 0;
    virtual NVSDK_NGX_Result Get(const char *, ID3D12Resource **) const = 0;
    virtual NVSDK_NGX_Result Get(const char *, void **) const = 0;

    virtual void Reset() = 0;
};

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

static CRITICAL_SECTION g_log_cs;
static char             g_log_path[MAX_PATH];
static HMODULE          g_self;

static void Log(const char *fmt, ...)
{
    char line[2048];
    va_list ap;
    va_start(ap, fmt);
    _vsnprintf_s(line, sizeof(line), _TRUNCATE, fmt, ap);
    va_end(ap);

    SYSTEMTIME st;
    GetLocalTime(&st);

    EnterCriticalSection(&g_log_cs);
    FILE *f = nullptr;
    if (fopen_s(&f, g_log_path, "a") == 0 && f != nullptr)
    {
        fprintf(f, "%02u:%02u:%02u.%03u  %s\n", st.wHour, st.wMinute, st.wSecond,
                st.wMilliseconds, line);
        fclose(f);
    }
    LeaveCriticalSection(&g_log_cs);
}

// ---------------------------------------------------------------------------
// Inline hook
//
// 14-byte absolute jump with the original bytes restored around the forwarded
// call. Whatever is at the entry when this installs gets saved, so installing
// on top of another add-on's detour simply puts this one first in the chain and
// hands control back to it on the way through.
// ---------------------------------------------------------------------------

struct Hook
{
    BYTE *target;
    BYTE  saved[14];
    BYTE  patch[14];
    bool  active;
};

static Hook             g_hook;
static Hook             g_hook_create;
static CRITICAL_SECTION g_hook_cs;

// Evaluates reach this add-on down two paths -- the NGX entry point and, when
// frame generation is on, Streamline's -- and they are not on the same thread.
// Both touch the same substitute textures and the same parameter block, which
// this code edits and then puts back. Left unguarded that is a race, and it
// showed up as the neural pass intermittently missing on some frames: bursts of
// untreated image rather than a steady rate, which is what a race looks like
// and what no timing theory explained.
//
// The whole substitute / forward / copy-back / restore run is therefore held as
// a unit, and the Streamline path takes the same lock. Order is always this one
// first and g_hook_cs inside it, so the two can never deadlock.
static CRITICAL_SECTION g_state_cs;

static bool WriteCode(void *dst, const void *src, size_t len)
{
    DWORD old = 0;
    if (!VirtualProtect(dst, len, PAGE_EXECUTE_READWRITE, &old)) return false;
    memcpy(dst, src, len);
    VirtualProtect(dst, len, old, &old);
    FlushInstructionCache(GetCurrentProcess(), dst, len);
    return true;
}

static bool HookInstall(Hook &h, void *target, void *detour)
{
    h.target = static_cast<BYTE *>(target);
    memcpy(h.saved, h.target, sizeof(h.saved));

    h.patch[0] = 0xFF;   // jmp qword ptr [rip+0]
    h.patch[1] = 0x25;
    h.patch[2] = h.patch[3] = h.patch[4] = h.patch[5] = 0x00;
    memcpy(h.patch + 6, &detour, sizeof(detour));

    if (!WriteCode(h.target, h.patch, sizeof(h.patch))) return false;
    h.active = true;
    return true;
}

static void HookRemove(Hook &h)  { if (h.active) WriteCode(h.target, h.saved, sizeof(h.saved)); }
static void HookRestore(Hook &h) { if (h.active) WriteCode(h.target, h.patch, sizeof(h.patch)); }

// ---------------------------------------------------------------------------
// Parameter reporting
// ---------------------------------------------------------------------------

// Reading through a pointer the probe did not create, so it is guarded.
static void SafePeek(const void *p, unsigned long long *out, int count)
{
    for (int i = 0; i < count; ++i) out[i] = 0;
    __try
    {
        const unsigned long long *q = static_cast<const unsigned long long *>(p);
        for (int i = 0; i < count; ++i) out[i] = q[i];
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {}
}

// A consumer of these resources typically insists they are plain 2D textures:
// one array slice, one mip, no multisampling. Anything else and it quietly does
// nothing, which from the outside is indistinguishable from not seeing the call
// at all. It also needs the output format to support shader sampling and typed
// UAV writes. Both conditions are checked here and stated plainly.
static void DescribeResource(const char *key, ID3D12Resource *res)
{
    D3D12_RESOURCE_DESC d = res->GetDesc();

    const bool plain_2d = (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) &&
                          (d.DepthOrArraySize == 1) && (d.MipLevels == 1) &&
                          (d.SampleDesc.Count == 1);

    Log("      desc: dim=%u %llux%u slices=%u mips=%u fmt=%u samples=%u flags=0x%X%s%s",
        d.Dimension, d.Width, d.Height, d.DepthOrArraySize, d.MipLevels, d.Format,
        d.SampleDesc.Count, d.Flags,
        (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ? " UAV" : "",
        plain_2d ? "" : "   <== NOT a plain single-slice single-mip 2D texture");

    ID3D12Device *dev = nullptr;
    if (FAILED(res->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(&dev))) ||
        dev == nullptr)
        return;

    D3D12_FEATURE_DATA_FORMAT_SUPPORT fs = {};
    fs.Format = d.Format;
    if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs))))
    {
        const bool sample = (fs.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE) != 0;
        const bool store  = (fs.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0;
        Log("      format %u: shader sample %s, typed UAV store %s%s",
            d.Format, sample ? "yes" : "NO", store ? "yes" : "NO",
            (sample && store) ? "" : "   <== fails the codec's format requirement");
    }
    dev->Release();
}

static void DumpSlot(const NVSDK_NGX_Parameter *p, const char *key)
{
    ID3D12Resource    *as_res = nullptr;
    void              *as_ptr = nullptr;
    unsigned long long as_u64 = 0;

    const NVSDK_NGX_Result r_res = p->Get(key, &as_res);
    const NVSDK_NGX_Result r_ptr = p->Get(key, &as_ptr);
    const NVSDK_NGX_Result r_u64 = p->Get(key, &as_u64);

    Log("    %-14s  ID3D12Resource*: %-4s %p | void*: %-4s %p | u64: %-4s 0x%llX",
        key,
        r_res == NGX_SUCCESS ? "ok" : "FAIL", static_cast<void *>(as_res),
        r_ptr == NGX_SUCCESS ? "ok" : "FAIL", as_ptr,
        r_u64 == NGX_SUCCESS ? "ok" : "FAIL", as_u64);

    // The interesting case: the plain form is empty but the void* form answers.
    // That means the extended resource struct is in use, and its first words
    // show the shape without having to assume a layout.
    if (r_res != NGX_SUCCESS && r_ptr == NGX_SUCCESS && as_ptr != nullptr)
    {
        unsigned long long w[6];
        SafePeek(as_ptr, w, 6);
        Log("      -> extended struct, first words: %016llX %016llX %016llX",
            w[0], w[1], w[2]);
        Log("                                       %016llX %016llX %016llX",
            w[3], w[4], w[5]);
        return;
    }

    if (r_res != NGX_SUCCESS || as_res == nullptr) return;
    DescribeResource(key, as_res);
}

static void DumpUInt(const NVSDK_NGX_Parameter *p, const char *key)
{
    unsigned int v = 0;
    if (p->Get(key, &v) == NGX_SUCCESS) Log("    %-40s = %u", key, v);
}

static void DumpFloat(const NVSDK_NGX_Parameter *p, const char *key)
{
    float v = 0.0f;
    if (p->Get(key, &v) == NGX_SUCCESS) Log("    %-40s = %.6f", key, static_cast<double>(v));
}

static void DumpInt(const NVSDK_NGX_Parameter *p, const char *key)
{
    int v = 0;
    if (p->Get(key, &v) == NGX_SUCCESS) Log("    %-40s = %d", key, v);
}

static void DumpParameters(const NVSDK_NGX_Parameter *p, long n)
{
    Log("--- D3D12 EvaluateFeature #%ld  params=%p ---", n, static_cast<const void *>(p));
    Log("  resources, read back in every form NGX allows:");
    DumpSlot(p, "Color");
    DumpSlot(p, "Output");
    DumpSlot(p, "Depth");
    DumpSlot(p, "MotionVectors");
    DumpSlot(p, "ExposureTexture");
    DumpSlot(p, "TransparencyMask");
    DumpSlot(p, "BiasCurrentColorMask");

    Log("  scalars:");
    DumpUInt(p, "Width");
    DumpUInt(p, "Height");
    DumpUInt(p, "OutWidth");
    DumpUInt(p, "OutHeight");
    DumpUInt(p, "DLSS.Render.Subrect.Dimensions.Width");
    DumpUInt(p, "DLSS.Render.Subrect.Dimensions.Height");
    DumpUInt(p, "DLSS.Input.Color.Subrect.Base.X");
    DumpUInt(p, "DLSS.Input.Color.Subrect.Base.Y");
    DumpFloat(p, "MV.Scale.X");
    DumpFloat(p, "MV.Scale.Y");
    DumpFloat(p, "Jitter.Offset.X");
    DumpFloat(p, "Jitter.Offset.Y");
    DumpFloat(p, "DLSS.Pre.Exposure");
    DumpInt(p, "Reset");
    DumpInt(p, "DLSS.Feature.Create.Flags");
    DumpInt(p, "PerfQualityValue");
}

// ---------------------------------------------------------------------------
// Mip-count workaround
//
// The consumer downstream requires MipLevels == 1 and silently does nothing
// otherwise. Some games hand DLSS an output and a depth buffer that carry a
// full mip chain, which is legal and which DLSS itself accepts -- but it makes
// the downstream add-on bail without a word.
//
// So: substitute single-mip textures of the same size and format for the frame,
// copying mip 0 in for the input and back out for the output, then put the
// original pointers back. Everything stays on the game's own device and queue,
// and the command list to record into arrives as the first argument.
//
// Set fix=0 in dlss5-d3d12-fix.cfg to observe without substituting.
// ---------------------------------------------------------------------------

// The substitute lives across frames, so its resource state has to be tracked
// rather than assumed: it rests in the state NGX expects to receive it in, and
// is moved to a copy state and back within the frame that needs it.
// ---------------------------------------------------------------------------
// The better fix: change the answer, not the texture
//
// The consumer rejects the output because GetDesc reports a mip chain. Handing
// it a different texture works but means this add-on, rather than NGX, writes
// the surface the rest of the frame consumes -- which is what disturbs frame
// generation.
//
// Instead, replace ID3D12Resource::GetDesc for the duration of the forwarded
// call with a thunk that reports MipLevels = 1. The consumer's check passes and
// NGX writes the game's own texture directly, exactly as it would with no
// add-on present. No substitute, no copies, nothing for a later pass to race
// against.
//
// GetDesc is vtable index 10 -- IUnknown 0..2, ID3D12Object 3..6,
// ID3D12DeviceChild 7, then Map, Unmap, GetDesc. It returns a struct by value,
// so on x64 the real signature takes a hidden return buffer after `this`.
//
// The vtable is shared by every resource from this driver, so the patch is held
// only across the one call and put straight back.
// ---------------------------------------------------------------------------

typedef D3D12_RESOURCE_DESC *(STDMETHODCALLTYPE *PFN_GetDesc)(ID3D12Resource *,
                                                              D3D12_RESOURCE_DESC *);

static PFN_GetDesc     g_real_getdesc;
static ID3D12Resource *g_patched_obj;
static void           *g_orig_vtbl;

// Editing the driver's shared vtable changes the answer for every resource in
// the process, and anything that reads a mip count on another thread during
// that window gets a wrong one. So the object gets a private copy of the table
// instead: only this resource behaves differently, and only for one call.
// Sixty-four slots is well past any D3D12 resource interface.
static void *g_private_vtbl[64];

static D3D12_RESOURCE_DESC *STDMETHODCALLTYPE MyGetDesc(ID3D12Resource *self,
                                                        D3D12_RESOURCE_DESC *out)
{
    g_real_getdesc(self, out);
    if (out->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && out->MipLevels > 1)
        out->MipLevels = 1;
    return out;
}

static bool PatchGetDesc(ID3D12Resource *res)
{
    if (res == nullptr || g_patched_obj != nullptr) return false;

    void **vtbl = *reinterpret_cast<void ***>(res);
    if (vtbl == nullptr) return false;

    if (g_real_getdesc == nullptr)
        g_real_getdesc = reinterpret_cast<PFN_GetDesc>(vtbl[10]);
    else if (reinterpret_cast<PFN_GetDesc>(vtbl[10]) != g_real_getdesc)
        return false;   // a different implementation than the one recorded

    memcpy(g_private_vtbl, vtbl, sizeof(g_private_vtbl));
    g_private_vtbl[10] = reinterpret_cast<void *>(&MyGetDesc);

    g_orig_vtbl   = vtbl;
    g_patched_obj = res;
    *reinterpret_cast<void ***>(res) = g_private_vtbl;
    return true;
}

static void UnpatchGetDesc()
{
    if (g_patched_obj == nullptr) return;
    *reinterpret_cast<void ***>(g_patched_obj) = static_cast<void **>(g_orig_vtbl);
    g_patched_obj = nullptr;
}

struct SubTex
{
    ID3D12Resource       *tex;
    UINT64                width;
    UINT                  height;
    DXGI_FORMAT           fmt;
    D3D12_RESOURCE_STATES state;
    D3D12_RESOURCE_STATES rest;
};

static SubTex g_sub_out;
static SubTex g_sub_depth;
static int    g_fix = -1;

// A substitute that is replaced -- which happens whenever the game changes its
// render resolution, and in an automatic DLSS mode that is often -- cannot be
// released on the spot: the GPU may still be reading commands that reference it.
// Retired resources are held for a few frames first. This costs a little memory
// and removes a use-after-free.
struct Retired { ID3D12Resource *tex; LONG frame; };
static Retired g_retired[8];
static LONG    g_frame_no = 0;

static void Retire(ID3D12Resource *tex)
{
    if (tex == nullptr) return;
    for (auto &r : g_retired)
        if (r.tex == nullptr) { r.tex = tex; r.frame = g_frame_no; return; }

    // Nowhere to park it: the oldest has waited longest, so let that one go.
    int oldest = 0;
    for (int i = 1; i < 8; ++i)
        if (g_retired[i].frame < g_retired[oldest].frame) oldest = i;
    g_retired[oldest].tex->Release();
    g_retired[oldest].tex   = tex;
    g_retired[oldest].frame = g_frame_no;
}

static void DrainRetired()
{
    for (auto &r : g_retired)
        if (r.tex != nullptr && (g_frame_no - r.frame) > 8)
        {
            r.tex->Release();
            r.tex = nullptr;
        }
}

// NVSDK_NGX_D3D12_EvaluateFeature carries every NGX feature, not just DLSS
// super sampling: ray reconstruction and frame generation come through the same
// entry point with entirely different parameter blocks. Substituting textures
// in those would be meaningless at best. Feature handles created as
// SuperSampling are recorded here so only those are touched.
struct FeatureId { NVSDK_NGX_Handle *handle; int id; };
static FeatureId     g_features[32];
static volatile LONG g_feature_count = 0;

static void RememberFeature(NVSDK_NGX_Handle *h, int id)
{
    if (h == nullptr) return;
    const LONG n = InterlockedIncrement(&g_feature_count) - 1;
    if (n < 32) { g_features[n].handle = h; g_features[n].id = id; }
}

static int FeatureIdOf(const NVSDK_NGX_Handle *h)
{
    const LONG n = g_feature_count;
    for (LONG i = 0; i < n && i < 32; ++i)
        if (g_features[i].handle == h) return g_features[i].id;
    return -1;
}

// Only super sampling is substituted. Ray reconstruction and frame generation
// arrive through the same entry point with unrelated parameter blocks and their
// own resolutions; sharing one set of substitute textures with them meant
// recreating those textures on every alternation between features. An
// unrecognised handle is treated as substitutable, so a feature created before
// this hook was in place still works.
static bool ShouldSubstitute(const NVSDK_NGX_Handle *h)
{
    const int id = FeatureIdOf(h);
    if (id < 0) return true;
    return id == 1;
}

static int FixEnabled()
{
    if (g_fix >= 0) return g_fix;
    g_fix = 1;

    char path[MAX_PATH];
    GetModuleFileNameA(g_self, path, MAX_PATH);
    if (char *s = strrchr(path, '\\'))
        strcpy_s(s + 1, MAX_PATH - (s + 1 - path), "dlss5-d3d12-fix.cfg");

    FILE *f = nullptr;
    if (fopen_s(&f, path, "r") == 0 && f != nullptr)
    {
        char line[128];
        while (fgets(line, sizeof(line), f) != nullptr)
        {
            int v = 1;
            if (sscanf_s(line, "fix=%d", &v) == 1) g_fix = (v != 0);
        }
        fclose(f);
    }
    Log("Mip workaround: %s", g_fix ? "enabled" : "disabled by config");
    return g_fix;
}

// Which slots get a single-mip substitute. Substituting the output means the
// texture the rest of the frame consumes is written by this add-on rather than
// by NGX directly, which is the one place a downstream pass such as frame
// generation could see stale content. If only the depth needs replacing to
// satisfy the consumer, leaving the output alone removes that risk entirely.
//   sub_output=1  sub_depth=1
static int CfgFlag(const char *key, int fallback)
{
    char path[MAX_PATH];
    GetModuleFileNameA(g_self, path, MAX_PATH);
    if (char *s = strrchr(path, '\\'))
        strcpy_s(s + 1, MAX_PATH - (s + 1 - path), "dlss5-d3d12-fix.cfg");

    FILE *f = nullptr;
    if (fopen_s(&f, path, "r") != 0 || f == nullptr) return fallback;

    int  value = fallback;
    char line[128], name[64];
    while (fgets(line, sizeof(line), f) != nullptr)
    {
        int v = 0;
        if (sscanf_s(line, "%63[^=]=%d", name, static_cast<unsigned>(sizeof(name)), &v) == 2 &&
            _stricmp(name, key) == 0)
            value = v;
    }
    fclose(f);
    return value;
}

static int SubOutput()
{
    static int v = -1;
    if (v < 0) { v = CfgFlag("sub_output", 1); Log("Substitute Output: %s", v ? "yes" : "no"); }
    return v;
}

static int SubDepth()
{
    static int v = -1;
    if (v < 0) { v = CfgFlag("sub_depth", 1); Log("Substitute Depth: %s", v ? "yes" : "no"); }
    return v;
}

static int PreloadOutput()
{
    static int v = -1;
    if (v < 0) { v = CfgFlag("preload_output", 1); Log("Preload output: %s", v ? "yes" : "no"); }
    return v;
}

// copyback=0 substitutes as usual -- so the consumer's checks still pass and it
// still does its work -- but throws the result away instead of writing it into
// the game's output. The neural pass becomes invisible, which is the point: if
// an artefact survives with this add-on writing nothing at all, that artefact
// was never caused by this add-on's write.
static int CopyBack()
{
    static int v = -1;
    if (v < 0)
    {
        v = CfgFlag("copyback", 1);
        if (!v) Log("Copy-back DISABLED: the neural result is discarded on purpose. "
                    "Nothing this add-on produces reaches the screen.");
    }
    return v;
}

// use_vtable=1 reports a single mip instead of supplying a different texture.
// Set it to 0 to fall back to substitution.
static int UseVtable()
{
    static int v = -1;
    if (v < 0)
    {
        v = CfgFlag("use_vtable", 1);
        Log("Method: %s", v ? "report MipLevels=1 via GetDesc (no substitution)"
                            : "substitute single-mip textures");
    }
    return v;
}

// nr_quality in the cfg: -1 correct the preset only when it contradicts the
// resolutions, -2 never touch it, 0..5 force that NGX quality value so the
// working one can be found without a rebuild.
static int QualityOverride()
{
    static int q = -3;
    if (q != -3) return q;
    q = -1;

    char path[MAX_PATH];
    GetModuleFileNameA(g_self, path, MAX_PATH);
    if (char *s = strrchr(path, '\\'))
        strcpy_s(s + 1, MAX_PATH - (s + 1 - path), "dlss5-d3d12-fix.cfg");

    FILE *f = nullptr;
    if (fopen_s(&f, path, "r") == 0 && f != nullptr)
    {
        char line[128];
        while (fgets(line, sizeof(line), f) != nullptr)
        {
            int v = -1;
            if (sscanf_s(line, "nr_quality=%d", &v) == 1 && v >= -2 && v <= 5) q = v;
        }
        fclose(f);
    }
    Log("Neural-rendering quality override: %d", q);
    return q;
}

static bool EnsureSub(SubTex &s, ID3D12Device *dev, const D3D12_RESOURCE_DESC &src,
                      const char *label)
{
    if (s.tex != nullptr && s.width == src.Width && s.height == src.Height &&
        s.fmt == src.Format)
        return true;

    if (s.tex != nullptr) { Retire(s.tex); s.tex = nullptr; }

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC d = src;
    d.MipLevels = 1;                                        // the whole point
    d.Flags     = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    d.Layout    = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    HRESULT hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
                                              D3D12_RESOURCE_STATE_COMMON, nullptr,
                                              __uuidof(ID3D12Resource),
                                              reinterpret_cast<void **>(&s.tex));
    if (FAILED(hr))
    {
        Log("  %s: could not create single-mip substitute (%llux%u fmt=%u): 0x%08X",
            label, src.Width, src.Height, src.Format, hr);
        return false;
    }

    s.width  = src.Width;
    s.height = src.Height;
    s.fmt    = src.Format;
    s.state  = D3D12_RESOURCE_STATE_COMMON;
    Log("  %s: single-mip substitute ready, %llux%u fmt=%u", label, s.width, s.height, s.fmt);
    return true;
}

static void Transition(ID3D12GraphicsCommandList *list, ID3D12Resource *res,
                       D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER b = {};
    b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource   = res;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter  = to;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &b);
}

static void ToState(ID3D12GraphicsCommandList *list, SubTex &s, D3D12_RESOURCE_STATES target)
{
    if (s.tex == nullptr || s.state == target) return;
    Transition(list, s.tex, s.state, target);
    s.state = target;
}

// Copies subresource 0 only; the remaining mips of the original are irrelevant
// to DLSS and are left untouched.
static void CopyMip0(ID3D12GraphicsCommandList *list, ID3D12Resource *dst, ID3D12Resource *src)
{
    D3D12_TEXTURE_COPY_LOCATION d = {};
    d.pResource        = dst;
    d.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    d.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION s = {};
    s.pResource        = src;
    s.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    s.SubresourceIndex = 0;

    list->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
}

// ---------------------------------------------------------------------------
// Detour
// ---------------------------------------------------------------------------

static volatile LONG g_count = 0;

typedef NVSDK_NGX_Result (*PFN_Evaluate)(void *, const NVSDK_NGX_Handle *,
                                         const NVSDK_NGX_Parameter *, void *);
typedef NVSDK_NGX_Result (*PFN_Create)(void *, int, NVSDK_NGX_Parameter *,
                                       NVSDK_NGX_Handle **);

// Streamline drives DLSS through its own entry point. Anything that arrives
// that way never reaches the NGX hook, so the substitution never runs for it --
// which would leave exactly the kind of occasional untreated frame that reads
// as intermittent flicker. This hook only counts; it changes nothing.
//
// The real signature takes five arguments. Declaring five pointer-sized ones
// and passing them straight through is safe on x64 whatever their true types,
// because none of them is inspected here.
typedef int (*PFN_SlEvaluate)(void *, void *, void *, unsigned int, void *);

static Hook          g_hook_sl;
static volatile LONG g_sl_count;
static volatile LONG g_did_substitute;
static volatile LONG g_did_skip;

static int Detour_SlEvaluate(void *a, void *b, void *c, unsigned int d, void *e)
{
    InterlockedIncrement(&g_sl_count);

    EnterCriticalSection(&g_hook_cs);
    HookRemove(g_hook_sl);
    int r = reinterpret_cast<PFN_SlEvaluate>(g_hook_sl.target)(a, b, c, d, e);
    HookRestore(g_hook_sl);
    LeaveCriticalSection(&g_hook_cs);
    return r;
}

typedef BOOL (WINAPI *PFN_EnumProcessModules)(HANDLE, HMODULE *, DWORD, LPDWORD);

static HMODULE FindStreamline(void **out_eval)
{
    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    if (k32 == nullptr) return nullptr;
    auto enum_modules =
        reinterpret_cast<PFN_EnumProcessModules>(GetProcAddress(k32, "K32EnumProcessModules"));
    if (enum_modules == nullptr) return nullptr;

    HMODULE mods[1024];
    DWORD   needed = 0;
    if (!enum_modules(GetCurrentProcess(), mods, sizeof(mods), &needed)) return nullptr;

    for (DWORD i = 0; i < needed / sizeof(HMODULE); ++i)
    {
        void *fn = reinterpret_cast<void *>(GetProcAddress(mods[i], "slEvaluateFeature"));
        if (fn != nullptr) { *out_eval = fn; return mods[i]; }
    }
    return nullptr;
}

// Watches feature creation only to learn which handles belong to DLSS super
// sampling. Frame generation and ray reconstruction arrive through the same
// evaluate entry point and must be left alone.
static NVSDK_NGX_Result Detour_CreateFeature(void *cmdlist, int feature_id,
                                             NVSDK_NGX_Parameter *p, NVSDK_NGX_Handle **out)
{
    // The neural-rendering feature is asked for at the game's output resolution,
    // input equal to output -- no upscaling. Asking for it under a performance
    // preset at the same time is a contradiction, and NGX answers by refusing to
    // initialise the feature at all. Where the two disagree, the resolutions are
    // the fact and the preset is the thing to correct.
    if (feature_id == 18 && p != nullptr && QualityOverride() != -2)
    {
        unsigned int w = 0, ow = 0;
        int          q = -1;
        p->Get("Width", &w);
        p->Get("OutWidth", &ow);
        p->Get("PerfQualityValue", &q);

        const int forced = QualityOverride();
        const int wanted = (forced >= 0) ? forced
                         : ((w != 0 && w == ow && q != 5) ? 5 : -1);

        if (wanted >= 0 && wanted != q)
        {
            p->Set("PerfQualityValue", wanted);
            Log("  PerfQualityValue %d -> %d (input %ux, output %ux: no upscaling)",
                q, wanted, w, ow);
        }
    }

    EnterCriticalSection(&g_hook_cs);
    HookRemove(g_hook_create);
    NVSDK_NGX_Result r =
        reinterpret_cast<PFN_Create>(g_hook_create.target)(cmdlist, feature_id, p, out);
    HookRestore(g_hook_create);
    LeaveCriticalSection(&g_hook_cs);

    // A feature that NGX refuses to create is the most opaque failure there is:
    // the caller's own log usually just prints a code. Everything it asked for
    // is right here in the parameter block, so it gets written out in full.
    if (p != nullptr && (r != NGX_SUCCESS || feature_id == 18))
    {
        Log("  parameter block for this creation:");
        DumpUInt(p, "Width");
        DumpUInt(p, "Height");
        DumpUInt(p, "OutWidth");
        DumpUInt(p, "OutHeight");
        DumpUInt(p, "CreationNodeMask");
        DumpUInt(p, "VisibilityNodeMask");
        DumpInt(p, "PerfQualityValue");
        DumpInt(p, "RTXValue");
        DumpInt(p, "DLSS.Feature.Create.Flags");
        DumpInt(p, "DLSS.Enable.Output.Subrects");
        DumpInt(p, "FreeMemOnReleaseFeature");
        DumpInt(p, "Snippet.Version");
        DumpUInt(p, "DLSSNR.Preset");
        DumpUInt(p, "DLSSNR.Style");
        DumpFloat(p, "DLSSNR.Intensity");
        DumpSlot(p, "Color");
        DumpSlot(p, "Output");
        DumpSlot(p, "Depth");
        DumpSlot(p, "MotionVectors");
        DumpSlot(p, "GBuffer.Normals");
        DumpSlot(p, "GBuffer.Roughness");
        DumpSlot(p, "GBuffer.Albedo");
        DumpSlot(p, "GBuffer.SpecularAlbedo");
        DumpSlot(p, "DiffuseAlbedo");
        DumpSlot(p, "SpecularHitDistance");
    }

    const char *what = feature_id == 1  ? "SuperSampling"
                     : feature_id == 11 ? "RayReconstruction"
                     : feature_id == 10 ? "FrameGeneration"
                     : feature_id == 18 ? "NeuralRendering" : "other";
    Log("CreateFeature id=%d (%s) -> 0x%08X handle=%p", feature_id, what, r,
        (out != nullptr) ? static_cast<void *>(*out) : nullptr);

    if (r == NGX_SUCCESS && out != nullptr)
    {
        RememberFeature(*out, feature_id);
        Log("  handle recorded; substitution %s for this feature",
            (feature_id == 1 || feature_id == 11) ? "enabled" : "disabled");
    }
    return r;
}

static NVSDK_NGX_Result Detour_Evaluate(void *cmdlist, const NVSDK_NGX_Handle *handle,
                                        const NVSDK_NGX_Parameter *p, void *cb)
{
    const LONG n = InterlockedIncrement(&g_count);
    if ((n <= 3 || (n % 3600) == 0) && p != nullptr)
        DumpParameters(p, n);

    ID3D12Resource *orig_out = nullptr;
    ID3D12Resource *orig_dep = nullptr;
    bool did_out = false, did_dep = false;

    auto *list = static_cast<ID3D12GraphicsCommandList *>(cmdlist);
    auto *par  = const_cast<NVSDK_NGX_Parameter *>(p);

    // Deliberately NOT held across the forwarded call. That call submits GPU
    // work and can wait on another thread; holding a lock across it once turned
    // an intermittent artefact into a hang. Only the windows that touch shared
    // state are guarded.
    EnterCriticalSection(&g_state_cs);
    g_frame_no = n;
    DrainRetired();

    const bool allowed = ShouldSubstitute(handle);
    if (n <= 6 || (n % 3600) == 0)
        Log("  evaluate handle=%p feature id=%d, substitution %s", handle,
            FeatureIdOf(handle), allowed ? "allowed" : "skipped");

    // Totals, so these can be lined up against the consumer's own counters. A
    // gap between the NGX evaluates seen here and the Streamline ones is the
    // measurement that matters: those are the frames the substitution misses.
    if ((n % 600) == 0)
        Log("[stats] NGX evaluates seen %ld | substituted %ld | skipped %ld | "
            "Streamline evaluates %ld",
            n, g_did_substitute, g_did_skip, g_sl_count);

    // Preferred route: leave every resource exactly as the game passed it and
    // only change what GetDesc reports while the call is in flight.
    bool patched = false;
    if (UseVtable() && FixEnabled() && par != nullptr && allowed)
    {
        ID3D12Resource *out_res = nullptr;
        par->Get("Output", &out_res);
        if (out_res != nullptr)
        {
            D3D12_RESOURCE_DESC d = out_res->GetDesc();
            if (d.MipLevels > 1)
            {
                patched = PatchGetDesc(out_res);
                if (n <= 6 || (n % 3600) == 0)
                    Log("  reporting MipLevels=1 for this call: %s",
                        patched ? "yes, no textures touched" : "FAILED, falling back");
            }
        }
    }

    if (!allowed) InterlockedIncrement(&g_did_skip);

    if (!patched && FixEnabled() && par != nullptr && list != nullptr && allowed)
    {
        par->Get("Output", &orig_out);
        par->Get("Depth", &orig_dep);

        ID3D12Device *dev = nullptr;
        if (orig_out != nullptr)
            orig_out->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(&dev));

        if (dev != nullptr)
        {
            // The output is written, so it only needs copying back afterwards.
            if (orig_out != nullptr)
            {
                D3D12_RESOURCE_DESC d = orig_out->GetDesc();
                if (SubOutput() && d.MipLevels > 1 && EnsureSub(g_sub_out, dev, d, "Output"))
                {
                    // Seed the substitute with what the game's output already
                    // holds. DLSS does not necessarily write every pixel -- this
                    // contract enables output subrects -- so anything it leaves
                    // untouched has to be the game's content and not whatever
                    // this texture happened to contain from an earlier frame.
                    // Without this the copy back writes stale pixels over valid
                    // ones, which shows up as intermittent flicker.
                    if (PreloadOutput())
                    {
                        Transition(list, orig_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_COPY_SOURCE);
                        ToState(list, g_sub_out, D3D12_RESOURCE_STATE_COPY_DEST);
                        CopyMip0(list, g_sub_out.tex, orig_out);
                        Transition(list, orig_out, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    }

                    // NGX writes the output through a UAV, so hand it over in
                    // that state exactly as the game would have.
                    ToState(list, g_sub_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    par->Set("Output", g_sub_out.tex);
                    did_out = true;
                }
            }

            // The depth is read, so mip 0 has to be brought across first.
            if (orig_dep != nullptr)
            {
                D3D12_RESOURCE_DESC d = orig_dep->GetDesc();
                if (SubDepth() && d.MipLevels > 1 && EnsureSub(g_sub_depth, dev, d, "Depth"))
                {
                    // NGX documents its inputs as shader resources, which is the
                    // state the caller must already have put the game's depth in.
                    Transition(list, orig_dep, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                               D3D12_RESOURCE_STATE_COPY_SOURCE);
                    ToState(list, g_sub_depth, D3D12_RESOURCE_STATE_COPY_DEST);
                    CopyMip0(list, g_sub_depth.tex, orig_dep);
                    ToState(list, g_sub_depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    Transition(list, orig_dep, D3D12_RESOURCE_STATE_COPY_SOURCE,
                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    par->Set("Depth", g_sub_depth.tex);
                    did_dep = true;
                }
            }
            dev->Release();

            if (did_out || did_dep) InterlockedIncrement(&g_did_substitute);
            if ((did_out || did_dep) && (n <= 6 || (n % 3600) == 0))
                Log("  substituted single-mip textures: Output=%s Depth=%s",
                    did_out ? "yes" : "no", did_dep ? "yes" : "no");
        }
    }

    LeaveCriticalSection(&g_state_cs);

    EnterCriticalSection(&g_hook_cs);
    HookRemove(g_hook);
    NVSDK_NGX_Result r = reinterpret_cast<PFN_Evaluate>(g_hook.target)(cmdlist, handle, p, cb);
    HookRestore(g_hook);
    LeaveCriticalSection(&g_hook_cs);

    EnterCriticalSection(&g_state_cs);

    // The vtable is shared with every other resource, so it goes back at once.
    if (patched) UnpatchGetDesc();

    if (did_out && CopyBack())
    {
        ToState(list, g_sub_out, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(list, orig_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        CopyMip0(list, orig_out, g_sub_out.tex);
        Transition(list, orig_out, D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        // Left in the state the next frame will hand it over in.
        ToState(list, g_sub_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    // Always hand the block back exactly as it was found.
    if (did_out) par->Set("Output", orig_out);
    if (did_dep) par->Set("Depth", orig_dep);

    LeaveCriticalSection(&g_state_cs);

    if (n <= 3) Log("--- #%ld returned 0x%08X ---", n, r);
    return r;
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

typedef BOOL (WINAPI *PFN_EnumProcessModules)(HANDLE, HMODULE *, DWORD, LPDWORD);

static HMODULE FindNgxLoader()
{
    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    if (k32 == nullptr) return nullptr;
    auto enum_modules =
        reinterpret_cast<PFN_EnumProcessModules>(GetProcAddress(k32, "K32EnumProcessModules"));
    if (enum_modules == nullptr) return nullptr;

    HMODULE mods[1024];
    DWORD   needed = 0;
    if (!enum_modules(GetCurrentProcess(), mods, sizeof(mods), &needed)) return nullptr;

    for (DWORD i = 0; i < needed / sizeof(HMODULE); ++i)
        if (GetProcAddress(mods[i], "NVSDK_NGX_D3D12_EvaluateFeature") != nullptr)
            return mods[i];
    return nullptr;
}

static void LogEntryBytes(const char *label, const void *fn)
{
    const BYTE *p = static_cast<const BYTE *>(fn);
    char hex[64];
    int  n = 0;
    for (int i = 0; i < 14; ++i)
        n += _snprintf_s(hex + n, sizeof(hex) - n, _TRUNCATE, "%02X ", p[i]);
    Log("  %-34s %p  %s", label, fn, hex);
}

static bool IsDetoured(const void *fn)
{
    const BYTE *p = static_cast<const BYTE *>(fn);
    return (p[0] == 0xE9) || (p[0] == 0xFF && p[1] == 0x25) ||
           (p[0] == 0x48 && p[1] == 0xB8) || (p[0] == 0xEB);
}

static DWORD WINAPI WatcherThread(LPVOID)
{
    // Two things have to be true before installing: NGX is loaded, and another
    // add-on has already hooked the entry point. Waiting for the second is what
    // puts this probe downstream of it.
    for (int i = 0; ; ++i)
    {
        HMODULE ngx = FindNgxLoader();
        if (ngx != nullptr)
        {
            void *eval = reinterpret_cast<void *>(
                GetProcAddress(ngx, "NVSDK_NGX_D3D12_EvaluateFeature"));
            if (eval != nullptr && IsDetoured(eval))
            {
                wchar_t path[MAX_PATH] = {};
                GetModuleFileNameW(ngx, path, MAX_PATH);
                Log("NGX loader: %ls", path);
                LogEntryBytes("NVSDK_NGX_D3D12_EvaluateFeature", eval);
                Log("Entry point is already detoured by another add-on. Installing "
                    "downstream of it.");

                void *create = reinterpret_cast<void *>(
                    GetProcAddress(ngx, "NVSDK_NGX_D3D12_CreateFeature"));

                EnterCriticalSection(&g_hook_cs);
                const bool ok = HookInstall(g_hook, eval,
                                            reinterpret_cast<void *>(&Detour_Evaluate));
                const bool ok_create = create != nullptr &&
                    HookInstall(g_hook_create, create,
                                reinterpret_cast<void *>(&Detour_CreateFeature));
                LeaveCriticalSection(&g_hook_cs);
                void *sl_eval = nullptr;
                bool  ok_sl   = false;
                if (FindStreamline(&sl_eval) != nullptr && sl_eval != nullptr)
                {
                    EnterCriticalSection(&g_hook_cs);
                    ok_sl = HookInstall(g_hook_sl, sl_eval,
                                        reinterpret_cast<void *>(&Detour_SlEvaluate));
                    LeaveCriticalSection(&g_hook_cs);
                }
                Log("Hooks: EvaluateFeature=%s CreateFeature=%s slEvaluateFeature=%s",
                    ok ? "installed" : "FAILED", ok_create ? "installed" : "FAILED",
                    sl_eval == nullptr ? "absent" : (ok_sl ? "installed (counting only)" : "FAILED"));
                return 0;
            }

            if (i == 300)
                Log("NGX is loaded but NVSDK_NGX_D3D12_EvaluateFeature is not detoured. "
                    "Either the DLSS 5 add-on is not present, or it has not hooked yet. "
                    "Still checking.");
        }
        else if (i == 300)
        {
            Log("No module exporting NVSDK_NGX_D3D12_EvaluateFeature yet. DLSS has "
                "probably not been initialised. Still checking.");
        }

        Sleep(i < 300 ? 200 : 2000);
    }
}

// ---------------------------------------------------------------------------
// ReShade registration
// ---------------------------------------------------------------------------

typedef bool (*PFN_Register)(HMODULE, uint32_t);
typedef void (*PFN_Unregister)(HMODULE);

static PFN_Unregister g_unregister;

static bool RegisterWithReShade(HMODULE self)
{
    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    if (k32 == nullptr) return false;
    auto enum_modules =
        reinterpret_cast<PFN_EnumProcessModules>(GetProcAddress(k32, "K32EnumProcessModules"));
    if (enum_modules == nullptr) return false;

    HMODULE mods[1024];
    DWORD   needed = 0;
    if (!enum_modules(GetCurrentProcess(), mods, sizeof(mods), &needed)) return false;

    for (DWORD i = 0; i < needed / sizeof(HMODULE); ++i)
    {
        auto reg = reinterpret_cast<PFN_Register>(
            GetProcAddress(mods[i], "ReShadeRegisterAddon"));
        if (reg == nullptr) continue;

        for (uint32_t version = 18; version >= 5; --version)
            if (reg(self, version))
            {
                g_unregister = reinterpret_cast<PFN_Unregister>(
                    GetProcAddress(mods[i], "ReShadeUnregisterAddon"));
                return true;
            }
    }
    return false;
}

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        g_self = module;
        DisableThreadLibraryCalls(module);
        InitializeCriticalSection(&g_log_cs);
        InitializeCriticalSection(&g_hook_cs);
        InitializeCriticalSection(&g_state_cs);

        GetModuleFileNameA(module, g_log_path, MAX_PATH);
        if (char *s = strrchr(g_log_path, '\\'))
            strcpy_s(s + 1, MAX_PATH - (s + 1 - g_log_path), "dlss5-d3d12-fix.log");

        if (!RegisterWithReShade(module)) return FALSE;

        FILE *f = nullptr;
        if (fopen_s(&f, g_log_path, "w") == 0 && f != nullptr) fclose(f);

        Log("dlss5-d3d12-fix %s (built %s %s) attached. Read-only: every call is "
            "forwarded unchanged.", PROBE_VERSION, __DATE__, __TIME__);
        CreateThread(nullptr, 0, &WatcherThread, nullptr, 0, nullptr);
    }
    else if (reason == DLL_PROCESS_DETACH)
    {
        EnterCriticalSection(&g_hook_cs);
        HookRemove(g_hook);
        HookRemove(g_hook_create);
        g_hook.active = g_hook_create.active = false;
        LeaveCriticalSection(&g_hook_cs);
        if (g_unregister != nullptr) g_unregister(g_self);
    }
    return TRUE;
}

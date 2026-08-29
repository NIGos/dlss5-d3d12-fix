# DLSS 5 D3D12 Mip Fix

A ReShade add-on that makes a DLSS 5 neural-rendering add-on work in a DirectX 12
game whose DLSS output carries a mip chain.

**Written for Resonance: A Plague Tale Legacy** — D3D12, Streamline, with Ray
Reconstruction and Frame Generation enabled. That is the game it was built
against and the only one it is known to fix.

Nothing in the code is specific to it: every size, format and resource is read
from the parameter block the caller passes. Whether other titles carry the same
defect is an open question, and the log answers it in the first second of play.

## The symptom

The DLSS 5 add-on's panel shows that it can see the game's DLSS calls, but never
produces a frame:

```
DLSSNR: STANDBY/FAILED
NGX hooks: creates 2 | evaluations 11960
Streamline direct fallback: 10885 attempts | 0 success
Successful NR frames: 0 | Guides: 0x0 | Output: 0x0
Latest NR NGX result: 0xBAD00005
```

Thousands of attempts, zero successes, and the guides never resolve.

## The cause

That add-on requires the DLSS output texture to have exactly one mip level. When
it doesn't, the add-on returns without doing anything and without logging
anything — indistinguishable from never seeing the call at all.

Some engines hand DLSS an output and a depth buffer with a full mip chain, which
is perfectly legal and which NGX itself accepts. Resonance's output has twelve.

## Why the check cannot simply be bypassed

Two ways of making the add-on accept a mipped texture were tried. Both hang the
GPU with `DXGI_ERROR_DEVICE_HUNG`, a few seconds after the neural pass starts:

- reporting a false `MipLevels` through a replaced `ID3D12Resource::GetDesc`
- patching the comparison out of the add-on's binary

The check is a guard, not a formality: the code behind it genuinely cannot
handle a mip chain. The only correct fix is to hand it a texture that really has
one mip.

## What it does

Around each DLSS evaluate:

1. create single-mip textures matching the game's output and depth
2. seed the output copy with what the game's output already holds — DLSS does
   not necessarily write every pixel, and this contract enables output subrects
3. put them in the NGX parameter block for the duration of the call
4. copy the result back into the game's output and restore the original pointers

Everything stays on the game's own device and queue, recorded into the command
list the evaluate is given. The DLSS 5 add-on is not modified.

Only the super-sampling feature is substituted. Ray reconstruction and frame
generation arrive through the same entry point with unrelated parameter blocks
and their own resolutions.

Where the output already has one mip, the add-on does nothing at all — it just
observes and logs. That makes it safe to try on any title.

## Requirements

In the game folder, alongside the game executable:

| File | Where from |
| --- | --- |
| `dxgi.dll` — ReShade 6.8+ **with add-on support** | reshade.me, full version |
| a DLSS 5 Neural Rendering ReShade add-on | its own author |
| `nvngx_dlssnr.dll` | shipped with that add-on |
| `dlss5-d3d12-fix.addon64` | this package |

Plus a D3D12 game with DLSS.

This add-on installs itself only once the other one has hooked the NGX entry
point, so it always sits downstream of it and sees exactly what it sees.

## Install

Drop `dlss5-d3d12-fix.addon64` next to ReShade. It writes its own
`dlss5-d3d12-fix.cfg` with working defaults. To remove it, delete the file.

## Configuration

`dlss5-d3d12-fix.cfg`, read at startup:

| Key | Default | Meaning |
| --- | --- | --- |
| `fix` | 1 | `0` observes and logs without substituting anything. |
| `sub_output` | 1 | Substitute the output. Required: the consumer's mip check is on this one. |
| `sub_depth` | 1 | Substitute the depth. |
| `preload_output` | 1 | Seed the substitute with the game's current output. Turning this off has crashed. |
| `copyback` | 1 | Copy the result back. `0` discards the whole DLSS output, not just the neural part — diagnostic only. |
| `use_vtable` | 0 | Report a false mip count instead of substituting. **Hangs the GPU.** Kept only to document that it was tried. |
| `nr_quality` | -2 | `-2` leave the consumer's quality preset alone. |

## Log

`dlss5-d3d12-fix.log` reports the NGX contract, a descriptor for every resource
with which of the consumer's requirements it meets, and running totals:

```
Output  desc: dim=3 3733x1600 slices=1 mips=12 fmt=26 samples=1 flags=0x5 UAV   <== NOT a plain single-slice single-mip 2D texture
        format 26: shader sample yes, typed UAV store yes
[stats] NGX evaluates seen 9000 | substituted 8991 | skipped 0 | Streamline evaluates 6902
```

If no resource is ever flagged, this add-on has nothing to do and the problem is
elsewhere. It is not a general remedy for DLSS 5 failing to start; it treats one
specific defect. The totals can be lined up against the consumer's own counters:
a gap between them is a set of frames the substitution missed.

## Threading

Evaluates arrive down two paths — the NGX entry point and, with frame generation
on, Streamline's — and not on the same thread. Both touch the same substitute
textures and the same parameter block. The windows that mutate shared state are
guarded; the forwarded call deliberately is not, because it submits GPU work and
can wait on another thread. Holding a lock across it turned an intermittent
artefact into a hang.

## Known limits

- The resource states the game leaves its textures in are assumed to be what NGX
  documents: inputs as shader resources, the output as a UAV. A game that
  differs would need different barriers. This is the most fragile assumption.
- Feature identification uses the standard NGX ids.
- Tested on one game, one GPU.
- Verbose logging is always on.

## Related

[dlss5-dx11-bridge](https://github.com/NIGos/dlss5-dx11-bridge) fixes a
different failure of the same add-on: a DirectX 11 game, where the D3D12 entry
points it hooks are never called at all. If the panel says it is waiting for the
game's DLSS rather than STANDBY/FAILED, that is the one to use.

## Building

Windows SDK and MSVC. No external dependencies; the ReShade add-on API is
reached through `GetProcAddress` and the NGX interfaces are declared inline.

From the `src` folder:

```
rc /nologo version.rc
cl /nologo /LD /EHsc /O2 /MT dlss5-d3d12-fix.cpp ^
   /link /OUT:dlss5-d3d12-fix.addon64 version.res kernel32.lib user32.lib
```

The version lives in `PROBE_VERSION` in the `.cpp` and in `version.rc`; they have
to stay in step.

## Reporting a problem

The first line of the log names the exact build. Post that plus the resource
descriptors and the last stats line — between them they identify the build, the
contract the game asked for, and whether the substitution was running.

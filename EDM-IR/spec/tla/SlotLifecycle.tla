--------------------------- MODULE SlotLifecycle ---------------------------
EXTENDS Naturals, FiniteSets

(***************************************************************************)
(* Bounded abstract protocol for an EDMIR staging slot.                    *)
(*                                                                         *)
(* Scope:                                                                  *)
(*   Free -> Producing -> Ready -> Consuming -> Released -> Producing      *)
(*                                                                         *)
(* The model covers compiler-owned publication, read, split completion,    *)
(* and reuse ordering.  It does not model CUTLASS's internal semaphore     *)
(* implementation, NCCL transport, CUDA scheduling, or floating point.     *)
(***************************************************************************)

CONSTANTS SlotCount, TileCount, SplitCount,
          UnsafeReuse, UnsafeRead, UnsafeRelease

ASSUME /\ SlotCount \in Nat \ {0}
       /\ TileCount \in Nat \ {0}
       /\ SplitCount \in Nat \ {0}
       /\ UnsafeReuse \in BOOLEAN
       /\ UnsafeRead \in BOOLEAN
       /\ UnsafeRelease \in BOOLEAN

Slots  == 1..SlotCount
Tiles  == 1..TileCount
Splits == 1..SplitCount
NoTile == 0

Phases == {"Free", "Producing", "Ready", "Consuming", "Released"}

VARIABLES phase,
          resident,
          epoch,
          nextTile,
          started,
          published,
          readStarted,
          doneSplits,
          completed

vars == <<phase, resident, epoch, nextTile, started, published,
          readStarted, doneSplits, completed>>

ResidentTiles == {resident[s] : s \in Slots} \ {NoTile}
LiveTiles     == started \ completed

Init ==
  /\ phase       = [s \in Slots |-> "Free"]
  /\ resident    = [s \in Slots |-> NoTile]
  /\ epoch       = [s \in Slots |-> 0]
  /\ nextTile    = 1
  /\ started     = {}
  /\ published   = {}
  /\ readStarted = {}
  /\ doneSplits  = [t \in Tiles |-> {}]
  /\ completed   = {}

Reusable(s) ==
  \/ phase[s] \in {"Free", "Released"}
  \/ UnsafeReuse /\ phase[s] \in {"Producing", "Ready", "Consuming"}

BeginProduce(s) ==
  /\ nextTile \in Tiles
  /\ Reusable(s)
  /\ phase'    = [phase EXCEPT ![s] = "Producing"]
  /\ resident' = [resident EXCEPT ![s] = nextTile]
  /\ epoch'    = [epoch EXCEPT ![s] = @ + 1]
  /\ nextTile' = nextTile + 1
  /\ started'  = started \cup {nextTile}
  /\ UNCHANGED <<published, readStarted, doneSplits, completed>>

Publish(s) ==
  /\ phase[s] = "Producing"
  /\ phase' = [phase EXCEPT ![s] = "Ready"]
  /\ published' = published \cup {resident[s]}
  /\ UNCHANGED <<resident, epoch, nextTile, started, readStarted,
                  doneSplits, completed>>

Readable(s) ==
  \/ phase[s] = "Ready"
  \/ UnsafeRead /\ phase[s] = "Producing"

StartConsume(s) ==
  /\ Readable(s)
  /\ phase' = [phase EXCEPT ![s] = "Consuming"]
  /\ readStarted' = readStarted \cup {resident[s]}
  /\ UNCHANGED <<resident, epoch, nextTile, started, published,
                  doneSplits, completed>>

FinishSplit(s, k) ==
  /\ phase[s] = "Consuming"
  /\ k \in Splits \ doneSplits[resident[s]]
  /\ doneSplits' =
       [doneSplits EXCEPT ![resident[s]] = @ \cup {k}]
  /\ UNCHANGED <<phase, resident, epoch, nextTile, started, published,
                  readStarted, completed>>

Releasable(s) ==
  \/ doneSplits[resident[s]] = Splits
  \/ UnsafeRelease /\ doneSplits[resident[s]] # Splits

Release(s) ==
  /\ phase[s] = "Consuming"
  /\ Releasable(s)
  /\ phase' = [phase EXCEPT ![s] = "Released"]
  /\ completed' = completed \cup {resident[s]}
  /\ UNCHANGED <<resident, epoch, nextTile, started, published,
                  readStarted, doneSplits>>

Done ==
  /\ completed = Tiles
  /\ UNCHANGED vars

Next ==
  \/ \E s \in Slots : BeginProduce(s)
  \/ \E s \in Slots : Publish(s)
  \/ \E s \in Slots : StartConsume(s)
  \/ \E s \in Slots, k \in Splits : FinishSplit(s, k)
  \/ \E s \in Slots : Release(s)
  \/ Done

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(***************************************************************************)
(* Type and structural invariants.                                         *)
(***************************************************************************)

TypeOK ==
  /\ phase \in [Slots -> Phases]
  /\ resident \in [Slots -> 0..TileCount]
  /\ epoch \in [Slots -> Nat]
  /\ nextTile \in 1..(TileCount + 1)
  /\ started \subseteq Tiles
  /\ published \subseteq started
  /\ readStarted \subseteq started
  /\ doneSplits \in [Tiles -> SUBSET Splits]
  /\ completed \subseteq started

SlotConsistency ==
  /\ \A s \in Slots :
       /\ (phase[s] = "Free") => resident[s] = NoTile
       /\ (phase[s] # "Free") => resident[s] \in started
       /\ (phase[s] = "Producing") =>
            doneSplits[resident[s]] = {}
       /\ (phase[s] = "Ready") =>
            /\ resident[s] \in published
            /\ doneSplits[resident[s]] = {}
       /\ (phase[s] = "Consuming") => resident[s] \in readStarted
       /\ (phase[s] = "Released") => resident[s] \in completed
  /\ \A t \in ResidentTiles :
       Cardinality({s \in Slots : resident[s] = t}) = 1

(***************************************************************************)
(* Stable assurance properties, matching ASSURANCE_MODEL.md.               *)
(***************************************************************************)

\* MEM-1: a consumer may start only after publication.
ReadAfterPublish == readStarted \subseteq published

\* MEM-2: every incomplete tile still has exactly one resident slot.
NoPrematureReuse == LiveTiles \subseteq ResidentTiles

\* RED-2: a released/completed tile contains every declared split partial.
CompleteReduction ==
  \A t \in completed : doneSplits[t] = Splits

\* Bounded liveness: weak fairness eventually completes every allocated tile.
EventualCompletion == <>(completed = Tiles)

=============================================================================

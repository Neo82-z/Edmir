------------------------- MODULE RemotePublication ------------------------
EXTENDS Naturals, FiniteSets

(***************************************************************************)
(* Bounded protocol for a mutable source buffer visible to remote readers. *)
(*                                                                         *)
(* Scope:                                                                  *)
(*   BeginWrite -> Publish -> Acquire -> Consume -> Retire -> BeginWrite   *)
(*                                                                         *)
(* Each publication carries an epoch.  The source may overwrite its buffer *)
(* only after every declared consumer has completed the prior epoch.       *)
(***************************************************************************)

CONSTANTS EpochCount, ConsumerCount,
          UnsafeAcquire, UnsafeConsume, UnsafeRetire, UnsafeOverwrite

ASSUME /\ EpochCount \in Nat \ {0}
       /\ ConsumerCount \in Nat \ {0}
       /\ UnsafeAcquire \in BOOLEAN
       /\ UnsafeConsume \in BOOLEAN
       /\ UnsafeRetire \in BOOLEAN
       /\ UnsafeOverwrite \in BOOLEAN

Epochs       == 1..EpochCount
Consumers    == 1..ConsumerCount
EpochReaders == Epochs \X Consumers
NoEpoch      == 0
Phases       == {"Empty", "Writing", "Published", "Retired"}

VARIABLES phase,
          current,
          nextEpoch,
          published,
          acquired,
          consumed,
          retired,
          overwritten

vars == <<phase, current, nextEpoch, published, acquired, consumed,
          retired, overwritten>>

AllConsumed(e) == \A c \in Consumers : <<e, c>> \in consumed

Init ==
  /\ phase       = "Empty"
  /\ current     = NoEpoch
  /\ nextEpoch   = 1
  /\ published   = {}
  /\ acquired    = {}
  /\ consumed    = {}
  /\ retired     = {}
  /\ overwritten = {}

Reusable ==
  \/ phase \in {"Empty", "Retired"}
  \/ UnsafeOverwrite /\ phase = "Published"

BeginWrite ==
  /\ nextEpoch \in Epochs
  /\ Reusable
  /\ phase' = "Writing"
  /\ current' = nextEpoch
  /\ nextEpoch' = nextEpoch + 1
  /\ overwritten' =
       IF current = NoEpoch THEN overwritten ELSE overwritten \cup {current}
  /\ UNCHANGED <<published, acquired, consumed, retired>>

Publish ==
  /\ phase = "Writing"
  /\ phase' = "Published"
  /\ published' = published \cup {current}
  /\ UNCHANGED <<current, nextEpoch, acquired, consumed, retired,
                  overwritten>>

Acquirable ==
  \/ phase = "Published"
  \/ UnsafeAcquire /\ phase = "Writing"

Acquire(c) ==
  /\ c \in Consumers
  /\ Acquirable
  /\ <<current, c>> \notin acquired
  /\ acquired' = acquired \cup {<<current, c>>}
  /\ UNCHANGED <<phase, current, nextEpoch, published, consumed, retired,
                  overwritten>>

Consumable(c) ==
  \/ <<current, c>> \in acquired
  \/ UnsafeConsume /\ current \in published

Consume(c) ==
  /\ c \in Consumers
  /\ phase = "Published"
  /\ <<current, c>> \notin consumed
  /\ Consumable(c)
  /\ consumed' = consumed \cup {<<current, c>>}
  /\ UNCHANGED <<phase, current, nextEpoch, published, acquired, retired,
                  overwritten>>

Retirable ==
  \/ AllConsumed(current)
  \/ UnsafeRetire

Retire ==
  /\ phase = "Published"
  /\ Retirable
  /\ phase' = "Retired"
  /\ retired' = retired \cup {current}
  /\ UNCHANGED <<current, nextEpoch, published, acquired, consumed,
                  overwritten>>

Done ==
  /\ retired = Epochs
  /\ UNCHANGED vars

Next ==
  \/ BeginWrite
  \/ Publish
  \/ \E c \in Consumers : Acquire(c)
  \/ \E c \in Consumers : Consume(c)
  \/ Retire
  \/ Done

Fairness ==
  /\ WF_vars(BeginWrite)
  /\ WF_vars(Publish)
  /\ \A c \in Consumers : WF_vars(Acquire(c))
  /\ \A c \in Consumers : WF_vars(Consume(c))
  /\ WF_vars(Retire)

Spec == Init /\ [][Next]_vars /\ Fairness

(***************************************************************************)
(* Structural invariants.                                                  *)
(***************************************************************************)

TypeOK ==
  /\ phase \in Phases
  /\ current \in 0..EpochCount
  /\ nextEpoch \in 1..(EpochCount + 1)
  /\ published \subseteq Epochs
  /\ acquired \subseteq EpochReaders
  /\ consumed \subseteq EpochReaders
  /\ retired \subseteq Epochs
  /\ overwritten \subseteq Epochs

PublicationConsistency ==
  /\ (phase = "Empty") => current = NoEpoch
  /\ (phase # "Empty") => current \in Epochs
  /\ (phase = "Writing") => current \notin published
  /\ (phase = "Published") => current \in published
  /\ (phase = "Retired") => current \in retired
  /\ retired \subseteq published
  /\ \A e \in published : e < nextEpoch

(***************************************************************************)
(* Stable publication properties.                                         *)
(***************************************************************************)

\* PUB-1: the matching epoch is published before a remote acquire.
AcquireAfterPublish ==
  \A e \in Epochs, c \in Consumers :
    <<e, c>> \in acquired => e \in published

\* PUB-2: an epoch is acquired before its data is consumed.
ConsumeAfterAcquire == consumed \subseteq acquired

\* PUB-3: retirement waits for every declared consumer.
CompleteConsumption == \A e \in retired : AllConsumed(e)

\* PUB-4: overwriting an epoch waits for every declared consumer.
NoOverwriteBeforeConsume == \A e \in overwritten : AllConsumed(e)

\* Bounded liveness under weak fairness of every protocol action.
EventualCompletion == <>(retired = Epochs)

=============================================================================

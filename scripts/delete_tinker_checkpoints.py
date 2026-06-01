"""Delete Tinker checkpoints for the current user.

Supports deleting all checkpoints, or only intermediate ones (keeping the
latest checkpoint per training run).
"""

from collections import defaultdict

import tinker

PAGE_SIZE = 100


def _collect_all_checkpoints(rest_client):
    """Paginate through all user checkpoints and return the full list."""
    all_checkpoints = []
    offset = 0

    while True:
        response = rest_client.list_user_checkpoints(
            limit=PAGE_SIZE, offset=offset
        ).result()

        checkpoints = response.checkpoints
        if not checkpoints:
            break

        all_checkpoints.extend(checkpoints)
        total_count = response.cursor.total_count if response.cursor else None

        print(
            f"Fetched {len(checkpoints)} checkpoints "
            f"(offset={offset}, total={total_count or 'unknown'})"
        )

        if total_count and offset + PAGE_SIZE >= total_count:
            break
        offset += PAGE_SIZE

    return all_checkpoints


def _get_run_id(cp):
    """Extract the training run ID from a checkpoint.

    Tries the direct attribute first, then parses from tinker_path.
    """
    run_id = getattr(cp, "training_run_id", None)
    if run_id:
        return run_id

    tinker_path = getattr(cp, "tinker_path", None)
    if tinker_path:
        try:
            parsed = tinker.types.ParsedCheckpointTinkerPath.from_tinker_path(
                tinker_path
            )
            return parsed.training_run_id
        except Exception:
            pass

    return "unknown"


def _identify_intermediate_checkpoints(checkpoints):
    """Return the subset of checkpoints that are NOT the latest per training run per type.

    Groups by (training_run_id, checkpoint_type), sorts each group by time
    descending, and marks every checkpoint except the newest one in each group
    as intermediate. This ensures both the latest training (weights) and
    sampler checkpoint are preserved for each run.
    """
    by_group = defaultdict(list)
    for cp in checkpoints:
        run_id = _get_run_id(cp)
        cp_type = getattr(cp, "checkpoint_type", None) or "unknown"
        by_group[(run_id, cp_type)].append(cp)

    intermediates = []
    kept = []
    for (run_id, cp_type), cps in by_group.items():
        # Sort newest-first by time; fall back to checkpoint_id string sort
        cps.sort(
            key=lambda c: (
                getattr(c, "time", None) or "",
                getattr(c, "checkpoint_id", "") or "",
            ),
            reverse=True,
        )
        kept.append(cps[0])
        intermediates.extend(cps[1:])

    return intermediates, kept


def _label(cp):
    """Human-readable label for a checkpoint."""
    tinker_path = getattr(cp, "tinker_path", None) or getattr(cp, "path", None)
    run_id = getattr(cp, "training_run_id", None)
    cp_id = getattr(cp, "checkpoint_id", None)

    if tinker_path:
        return tinker_path
    elif run_id and cp_id:
        return f"{run_id}/{cp_id}"
    else:
        return repr(cp)


def _delete_checkpoint(rest_client, cp):
    """Delete a single checkpoint, return True on success."""
    tinker_path = getattr(cp, "tinker_path", None) or getattr(cp, "path", None)
    run_id = getattr(cp, "training_run_id", None)
    cp_id = getattr(cp, "checkpoint_id", None)

    if tinker_path:
        rest_client.delete_checkpoint_from_tinker_path(tinker_path).result()
    elif run_id and cp_id:
        rest_client.delete_checkpoint(run_id, cp_id).result()
    else:
        return False
    return True


def _is_protected(cp, keep_set):
    """Check if a checkpoint matches any entry in the keep set.

    Matches against tinker_path, checkpoint_id, or run_id/checkpoint_id.
    """
    if not keep_set:
        return False

    tinker_path = getattr(cp, "tinker_path", None) or ""
    cp_id = getattr(cp, "checkpoint_id", None) or ""
    run_id = _get_run_id(cp)
    combo = f"{run_id}/{cp_id}" if run_id and cp_id else ""

    for k in keep_set:
        if k and (k == tinker_path or k == cp_id or k == combo):
            return True
    return False


def delete_checkpoints(
    dry_run: bool = True, intermediate_only: bool = False, keep: list[str] | None = None
):
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    keep_set = set(keep) if keep else set()

    if keep_set:
        print(f"Protected checkpoints (will NOT be deleted): {keep_set}")

    if intermediate_only:
        # We need all checkpoints upfront to figure out which are intermediate
        print("Collecting all checkpoints to identify intermediates...")
        all_cps = _collect_all_checkpoints(rest_client)
        if not all_cps:
            print("No checkpoints found.")
            return

        # Print fields for debugging on first run
        sample = all_cps[0]
        fields = (
            sample.model_fields.keys()
            if hasattr(sample, "model_fields")
            else vars(sample).keys()
        )
        print(f"Checkpoint fields: {list(fields)}")

        intermediates, kept = _identify_intermediate_checkpoints(all_cps)

        print(
            f"\n{len(all_cps)} total checkpoints across "
            f"{len(set(_get_run_id(c) for c in all_cps))} training run(s)"
        )
        print(f"  Keeping (latest per run per type): {len(kept)}")
        print(f"  Intermediate (to delete): {len(intermediates)}")

        if not intermediates:
            print("\nNo intermediate checkpoints to delete.")
            return

        total_deleted = 0
        total_protected = 0
        for cp in intermediates:
            lab = _label(cp)
            if _is_protected(cp, keep_set):
                print(f"  [PROTECTED] Skipping: {lab}")
                total_protected += 1
                continue
            if dry_run:
                print(f"  [DRY RUN] Would delete: {lab}")
            else:
                try:
                    if _delete_checkpoint(rest_client, cp):
                        print(f"  Deleted: {lab}")
                        total_deleted += 1
                    else:
                        print(f"  SKIP (no path or IDs): {lab}")
                except Exception as e:
                    print(f"  FAILED to delete {lab}: {e}")

        if dry_run:
            print(
                f"\nDry run complete. Would have deleted "
                f"{len(intermediates) - total_protected} intermediate checkpoint(s)"
                f"{f', skipped {total_protected} protected' if total_protected else ''}."
            )
        else:
            print(
                f"\nDone. Deleted {total_deleted} intermediate checkpoint(s)"
                f"{f', skipped {total_protected} protected' if total_protected else ''}."
            )

    else:
        # Original delete-all behaviour with streaming pagination
        offset = 0
        total_deleted = 0
        protected_count = 0  # track to offset past protected checkpoints

        while True:
            response = rest_client.list_user_checkpoints(
                limit=PAGE_SIZE, offset=offset
            ).result()

            checkpoints = response.checkpoints
            if not checkpoints:
                break

            total_count = response.cursor.total_count if response.cursor else "unknown"
            print(
                f"Fetched {len(checkpoints)} checkpoints "
                f"(offset={offset}, total={total_count})"
            )

            if offset == 0 and checkpoints:
                sample = checkpoints[0]
                fields = (
                    sample.model_fields.keys()
                    if hasattr(sample, "model_fields")
                    else vars(sample).keys()
                )
                print(f"  Checkpoint fields: {list(fields)}")

            batch_protected = 0
            for cp in checkpoints:
                lab = _label(cp)
                if _is_protected(cp, keep_set):
                    print(f"  [PROTECTED] Skipping: {lab}")
                    batch_protected += 1
                    continue
                if dry_run:
                    print(f"  [DRY RUN] Would delete: {lab}")
                else:
                    try:
                        if _delete_checkpoint(rest_client, cp):
                            print(f"  Deleted: {lab}")
                            total_deleted += 1
                        else:
                            print(f"  SKIP (no path or IDs): {lab}")
                    except Exception as e:
                        print(f"  FAILED to delete {lab}: {e}")

            if dry_run:
                offset += PAGE_SIZE
                if response.cursor and offset >= response.cursor.total_count:
                    break
            else:
                # Protected checkpoints stay in the list, so we must
                # offset past them to avoid re-fetching the same ones.
                protected_count += batch_protected
                offset = protected_count
                if len(checkpoints) < PAGE_SIZE:
                    break

        if dry_run:
            print("\nDry run complete. No checkpoints were deleted.")
        else:
            print(f"\nDone. Deleted {total_deleted} checkpoint(s).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Delete Tinker checkpoints for the current user."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--intermediate-only",
        action="store_true",
        help=(
            "Only delete intermediate checkpoints, keeping the latest "
            "checkpoint per training run."
        ),
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        metavar="PATH_OR_ID",
        help=(
            "Checkpoint to protect from deletion. Can be a tinker path "
            "(e.g. tinker://run-id/weights/0042), a checkpoint_id, or "
            "run_id/checkpoint_id. Repeatable."
        ),
    )
    args = parser.parse_args()

    delete_checkpoints(
        dry_run=not args.confirm,
        intermediate_only=args.intermediate_only,
        keep=args.keep,
    )

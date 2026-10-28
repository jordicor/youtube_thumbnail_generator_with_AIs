"""One-time, transactional migration from the old reference selectors.

Run only during offline activation. Originals are never rewritten or removed.
"""
from pathlib import Path

import orjson

from services.reference_service import REFERENCE_SCHEMA, ReferenceService, VisualReference


async def migrate_visual_references(db):
    async with db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='visual_references'") as cursor:
        if await cursor.fetchone():
            return
    async with db.execute("PRAGMA table_info(clusters)") as cursor:
        columns = {row[1] for row in await cursor.fetchall()}
    async with db.execute("PRAGMA table_info(videos)") as cursor:
        video_columns = {row[1] for row in await cursor.fetchall()}
    await db.execute("SAVEPOINT visual_reference_migration")
    try:
        for statement in REFERENCE_SCHEMA.split(";"):
            if statement.strip():
                await db.execute(statement)
        if "reference_format" not in columns:
            await db.execute("ALTER TABLE clusters ADD COLUMN reference_format TEXT NOT NULL DEFAULT 'photos'")
        if "reference_revision" not in video_columns:
            await db.execute("ALTER TABLE videos ADD COLUMN reference_revision INTEGER NOT NULL DEFAULT 0")
        if "environment_mode" not in video_columns:
            await db.execute("ALTER TABLE videos ADD COLUMN environment_mode TEXT NOT NULL DEFAULT 'same'")
        service = ReferenceService(db)
        fields = ["id", "video_id"] + [name if name in columns else f"NULL AS {name}"
                                     for name in ("face_sheet", "face_sheet_enabled", "body_reference", "environment_reference")]
        async with db.execute(f"SELECT {', '.join(fields)} FROM clusters ORDER BY video_id, CASE WHEN view_mode='person' THEN 0 ELSE 1 END, cluster_index, id") as cursor:
            people = await cursor.fetchall()
        environments = {}
        automatic_settings = {}
        for person_id, video_id, sheet, enabled, body, setting in people:
            async with db.execute("""
                SELECT vf.frame_path, cfa.is_reference FROM cluster_frame_assignments cfa
                JOIN video_frames vf ON vf.id = cfa.frame_id WHERE cfa.cluster_id = ?
                ORDER BY cfa.is_reference DESC, cfa.reference_order, vf.quality_score DESC, vf.id
            """, [person_id]) as cursor:
                rows = await cursor.fetchall()
            names = [Path(path).name for path, selected in rows if selected]
            # Old generators used quality-ranked members only when no explicit photos existed.
            if not names and rows:
                names = [Path(rows[0][0]).name]
            views = orjson.loads(sheet) if sheet else {}
            refs = {}
            for name in names:
                refs[name] = VisualReference(role="face", person_id=person_id, source_type="frame",
                                             source=name, primary=not refs).model_dump()
            for view, name in views.items():
                if not name:
                    continue  # An empty manual slot remains empty; no automatic filling.
                if name not in refs:
                    refs[name] = VisualReference(role="face", person_id=person_id, source_type="frame",
                                                 source=name, primary=not refs).model_dump()
                if view in ("front", "profile"):
                    refs[name]["view"] = view
                elif view in ("three_quarter_a", "three_quarter_b"):
                    refs[name]["view"] = "three_quarter"
                elif view in ("smiling", "mouth_open"):
                    refs[name]["expression"] = view
            if enabled:
                # Freeze active automatic collage cells once, rather than retain another selector.
                output = await service.output_dir(video_id)
                if (output / "faces.json").is_file():
                    from services.face_index import load_face_index, match_faces, stored_centroid, is_grayscale_face
                    from services.face_sheet import automatic_views, resolve_views, view_candidates
                    index = load_face_index(output / "faces.json")
                    members = [Path(row[0]).name for row in rows]
                    centroid = stored_centroid(await service.analysis.centroid_blob(person_id), index, members, person_id)
                    matches = match_faces(index, centroid, members)
                    cells = resolve_views(automatic_views(view_candidates(matches), lambda name: not is_grayscale_face(output / "frames" / name, matches[name].box)), views, set(members))
                    for view, name in cells:
                        if name not in refs:
                            refs[name] = VisualReference(role="face", person_id=person_id, source_type="frame",
                                                         source=name, primary=not refs).model_dump()
                await db.execute("UPDATE clusters SET reference_format='collage' WHERE id=?", [person_id])
            selected = list(refs.values())
            if body:
                selected.append(VisualReference(role="body", person_id=person_id, source_type="frame", source=body).model_dump())
            elif body is None and rows:
                output = await service.output_dir(video_id)
                if (output / "body_references.json").is_file() and (output / "faces.json").is_file():
                    from services.body_references import load_body_index, person_body_candidates
                    from services.face_index import load_face_index, match_faces, stored_centroid, is_grayscale_face
                    index = load_face_index(output / "faces.json")
                    members = [Path(row[0]).name for row in rows]
                    centroid = stored_centroid(await service.analysis.centroid_blob(person_id), index, members, person_id)
                    candidates = person_body_candidates(load_body_index(output), match_faces(index, centroid, members))["candidates"]
                    if candidates:
                        candidate = candidates[0]
                        selected.append(VisualReference(
                            role="body", person_id=person_id, source_type="frame", source=candidate["filename"],
                            crop=candidate["box"], coverage=candidate["coverage"], posture=candidate["posture"],
                            identity_confirmed=True,
                        ).model_dump())
            await service.insert(video_id, selected)
            # NULL was an automatic choice, not a manual selection per scene.
            # Freeze one initial video context, preserving every explicit setting separately.
            if video_id not in automatic_settings:
                automatic_settings[video_id] = (names[0] if names else (Path(rows[0][0]).name if rows else None)) if setting is None else None
            if setting:
                environments.setdefault(video_id, {})[setting] = VisualReference(
                    role="environment", source_type="frame", source=setting,
                ).model_dump()
        for video_id, setting in automatic_settings.items():
            if video_id not in environments and setting:
                environments[video_id] = {setting: VisualReference(
                    role="environment", source_type="frame", source=setting,
                ).model_dump()}
        for video_id, references in environments.items():
            await service.insert(video_id, list(references.values()))
            await db.execute("UPDATE videos SET environment_mode=? WHERE id=?",
                             ["alternatives" if len(references) > 1 else "same", video_id])
        for name in ("face_sheet", "face_sheet_enabled", "body_reference", "environment_reference"):
            if name in columns:
                await db.execute(f"ALTER TABLE clusters DROP COLUMN {name}")
        await db.execute("DROP INDEX IF EXISTS idx_assignments_reference")
        await db.execute("ALTER TABLE cluster_frame_assignments DROP COLUMN is_reference")
        await db.execute("ALTER TABLE cluster_frame_assignments DROP COLUMN reference_order")
    except BaseException:
        await db.execute("ROLLBACK TO visual_reference_migration")
        raise
    finally:
        await db.execute("RELEASE visual_reference_migration")
    await db.commit()

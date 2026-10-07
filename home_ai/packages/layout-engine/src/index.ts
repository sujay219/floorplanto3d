import type { Scene, SceneObject } from '@home-ai/scene-schema';

export interface PlacementRequest {
  object: SceneObject;
  roomId: string;
  preferredPosition?: { x: number; y: number; z: number };
}

export interface PlacementResult {
  accepted: boolean;
  reason?: string;
  object: SceneObject;
}

export interface CollisionResult {
  collides: boolean;
  overlaps: string[];
  message?: string;
}

export interface LayoutValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface LayoutEngine {
  validate(scene: Scene): LayoutValidationResult;
  place(scene: Scene, request: PlacementRequest): PlacementResult;
}

export class DefaultLayoutEngine implements LayoutEngine {
  validate(scene: Scene): LayoutValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!scene) {
      errors.push('Scene is required.');
    }

    if (scene && scene.units !== 'mm') {
      warnings.push('Scene units are not millimeters.');
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  }

  place(scene: Scene, request: PlacementRequest): PlacementResult {
    const validation = this.validate(scene);

    if (!validation.valid) {
      return {
        accepted: false,
        reason: validation.errors.join(', '),
        object: request.object,
      };
    }

    return {
      accepted: true,
      object: {
        ...request.object,
        position: request.preferredPosition ?? request.object.position,
      },
    };
  }
}

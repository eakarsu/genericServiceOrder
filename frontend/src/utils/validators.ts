import { z } from 'zod';

export const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
});

export const registerSchema = z.object({
  name: z.string().min(1, 'Name is required').max(255),
  email: z.string().email('Invalid email address'),
  password: z.string().min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Must contain an uppercase letter')
    .regex(/[a-z]/, 'Must contain a lowercase letter')
    .regex(/\d/, 'Must contain a digit')
    .regex(/[!@#$%^&*(),.?":{}|<>]/, 'Must contain a special character'),
  phone: z.string().optional(),
});

export const forgotPasswordSchema = z.object({
  email: z.string().email('Invalid email address'),
});

export const resetPasswordSchema = z.object({
  new_password: z.string().min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Must contain an uppercase letter')
    .regex(/[a-z]/, 'Must contain a lowercase letter')
    .regex(/\d/, 'Must contain a digit')
    .regex(/[!@#$%^&*(),.?":{}|<>]/, 'Must contain a special character'),
});

export const changePasswordSchema = z.object({
  current_password: z.string().min(1, 'Current password is required'),
  new_password: z.string().min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Must contain an uppercase letter')
    .regex(/[a-z]/, 'Must contain a lowercase letter')
    .regex(/\d/, 'Must contain a digit')
    .regex(/[!@#$%^&*(),.?":{}|<>]/, 'Must contain a special character'),
});

export const profileSchema = z.object({
  name: z.string().min(1, 'Name is required').max(255),
  phone: z.string().optional(),
});

export const orderEditSchema = z.object({
  customer_name: z.string().min(1).optional(),
  customer_phone: z.string().optional(),
  customer_email: z.string().email().optional().or(z.literal('')),
  status: z.string().optional(),
  notes: z.string().optional(),
});

export type LoginForm = z.infer<typeof loginSchema>;
export type RegisterForm = z.infer<typeof registerSchema>;
export type ForgotPasswordForm = z.infer<typeof forgotPasswordSchema>;
export type ResetPasswordForm = z.infer<typeof resetPasswordSchema>;
export type ChangePasswordForm = z.infer<typeof changePasswordSchema>;
export type ProfileForm = z.infer<typeof profileSchema>;
export type OrderEditForm = z.infer<typeof orderEditSchema>;
